"""Tests for Canvas connector."""

from datetime import datetime
from datetime import timezone
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from onyx.configs.constants import DocumentSource
from onyx.connectors.canvas.connector import CanvasApiClient
from onyx.connectors.canvas.connector import CanvasClientRequestFailedError
from onyx.connectors.canvas.connector import CanvasConnector
from onyx.connectors.exceptions import ConnectorValidationError
from onyx.connectors.exceptions import CredentialExpiredError
from onyx.connectors.exceptions import UnexpectedValidationError
from onyx.connectors.models import ConnectorMissingCredentialError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FAKE_BASE_URL = "https://myschool.instructure.com"
FAKE_TOKEN = "fake-canvas-token"


def _build_connector(base_url: str = FAKE_BASE_URL) -> CanvasConnector:
    connector = CanvasConnector(canvas_base_url=base_url)
    connector.load_credentials({"canvas_access_token": FAKE_TOKEN})
    return connector


def _mock_course(
    course_id: int = 1,
    name: str = "Intro to CS",
    course_code: str = "CS101",
) -> dict:
    return {
        "id": course_id,
        "name": name,
        "course_code": course_code,
        "created_at": "2025-01-01T00:00:00Z",
        "workflow_state": "available",
    }


def _mock_page(
    page_id: int = 10,
    title: str = "Syllabus",
    course_id: int = 1,
    updated_at: str = "2025-06-01T12:00:00Z",
) -> dict:
    return {
        "page_id": page_id,
        "url": "syllabus",
        "title": title,
        "body": "<p>Welcome to the course</p>",
        "created_at": "2025-01-15T00:00:00Z",
        "updated_at": updated_at,
    }


def _mock_assignment(
    assignment_id: int = 20,
    name: str = "Homework 1",
    course_id: int = 1,
    updated_at: str = "2025-06-01T12:00:00Z",
) -> dict:
    return {
        "id": assignment_id,
        "name": name,
        "description": "<p>Solve these problems</p>",
        "html_url": f"{FAKE_BASE_URL}/courses/{course_id}/assignments/{assignment_id}",
        "course_id": course_id,
        "created_at": "2025-01-20T00:00:00Z",
        "updated_at": updated_at,
        "due_at": "2025-02-01T23:59:00Z",
    }


def _mock_announcement(
    announcement_id: int = 30,
    title: str = "Class Cancelled",
    course_id: int = 1,
    posted_at: str = "2025-06-01T12:00:00Z",
) -> dict:
    return {
        "id": announcement_id,
        "title": title,
        "message": "<p>No class today</p>",
        "html_url": f"{FAKE_BASE_URL}/courses/{course_id}/discussion_topics/{announcement_id}",
        "posted_at": posted_at,
    }


# ---------------------------------------------------------------------------
# CanvasApiClient tests
# ---------------------------------------------------------------------------


class TestCanvasApiClient:
    def test_build_url(self) -> None:
        client = CanvasApiClient(
            bearer_token=FAKE_TOKEN,
            canvas_base_url=FAKE_BASE_URL,
        )
        assert client._build_url("courses") == (
            f"{FAKE_BASE_URL}/api/v1/courses"
        )

    def test_build_url_strips_trailing_slash(self) -> None:
        client = CanvasApiClient(
            bearer_token=FAKE_TOKEN,
            canvas_base_url=f"{FAKE_BASE_URL}/",
        )
        assert client._build_url("courses") == (
            f"{FAKE_BASE_URL}/api/v1/courses"
        )

    def test_build_headers(self) -> None:
        client = CanvasApiClient(
            bearer_token=FAKE_TOKEN,
            canvas_base_url=FAKE_BASE_URL,
        )
        assert client._build_headers() == {
            "Authorization": f"Bearer {FAKE_TOKEN}"
        }

    @patch("onyx.connectors.canvas.connector.rl_requests")
    def test_get_raises_on_error_status(self, mock_requests: MagicMock) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 403
        mock_response.reason = "Forbidden"
        mock_response.json.return_value = {}
        mock_requests.get.return_value = mock_response

        client = CanvasApiClient(
            bearer_token=FAKE_TOKEN,
            canvas_base_url=FAKE_BASE_URL,
        )
        with pytest.raises(CanvasClientRequestFailedError) as exc_info:
            client.get("courses")
        assert exc_info.value.status_code == 403


# ---------------------------------------------------------------------------
# CanvasConnector — credential loading
# ---------------------------------------------------------------------------


class TestLoadCredentials:
    @patch("onyx.connectors.canvas.connector.rl_requests")
    def test_load_credentials_success(self, mock_requests: MagicMock) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = [_mock_course()]
        mock_requests.get.return_value = mock_response

        connector = CanvasConnector(canvas_base_url=FAKE_BASE_URL)
        result = connector.load_credentials({"canvas_access_token": FAKE_TOKEN})
        assert result is None
        assert connector._canvas_client is not None

    def test_canvas_client_raises_without_credentials(self) -> None:
        connector = CanvasConnector(canvas_base_url=FAKE_BASE_URL)
        with pytest.raises(ConnectorMissingCredentialError):
            _ = connector.canvas_client

    @patch("onyx.connectors.canvas.connector.rl_requests")
    def test_load_credentials_invalid_token(self, mock_requests: MagicMock) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.reason = "Unauthorized"
        mock_response.json.return_value = {}
        mock_requests.get.return_value = mock_response

        connector = CanvasConnector(canvas_base_url=FAKE_BASE_URL)
        with pytest.raises(ConnectorMissingCredentialError, match="Invalid Canvas API token"):
            connector.load_credentials({"canvas_access_token": "bad-token"})


# ---------------------------------------------------------------------------
# CanvasConnector — document conversion
# ---------------------------------------------------------------------------


class TestDocumentConversion:
    @patch("onyx.connectors.canvas.connector.rl_requests")
    def setup_method(self, method: object, mock_requests: MagicMock = MagicMock()) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = [_mock_course()]

        # Patch at module level for load_credentials call
        with patch("onyx.connectors.canvas.connector.rl_requests") as mock_req:
            mock_req.get.return_value = mock_response
            self.connector = _build_connector()

    def test_convert_page_to_document(self) -> None:
        from onyx.connectors.canvas.connector import CanvasPage

        page = CanvasPage(
            page_id=10,
            url="syllabus",
            title="Syllabus",
            body="<p>Welcome</p>",
            created_at="2025-01-15T00:00:00Z",
            updated_at="2025-06-01T12:00:00Z",
            course_id=1,
        )
        doc = self.connector._convert_page_to_document(page)

        assert doc.id == "canvas-page-1-10"
        assert doc.source == DocumentSource.CANVAS
        assert doc.semantic_identifier == "Syllabus"
        assert doc.metadata == {"course_id": "1"}
        assert f"{FAKE_BASE_URL}/courses/1/pages/syllabus" in doc.sections[0].link
        assert doc.doc_updated_at == datetime(2025, 6, 1, 12, 0, tzinfo=timezone.utc)

    def test_convert_page_without_body(self) -> None:
        from onyx.connectors.canvas.connector import CanvasPage

        page = CanvasPage(
            page_id=11,
            url="empty-page",
            title="Empty Page",
            body=None,
            created_at="2025-01-15T00:00:00Z",
            updated_at="2025-06-01T12:00:00Z",
            course_id=1,
        )
        doc = self.connector._convert_page_to_document(page)

        # Body is None, so text should only have title and link
        section_text = doc.sections[0].text
        assert "Empty Page" in section_text
        assert "<p>" not in section_text

    def test_convert_assignment_to_document(self) -> None:
        from onyx.connectors.canvas.connector import CanvasAssignment

        assignment = CanvasAssignment(
            id=20,
            name="Homework 1",
            description="<p>Solve these</p>",
            html_url=f"{FAKE_BASE_URL}/courses/1/assignments/20",
            course_id=1,
            created_at="2025-01-20T00:00:00Z",
            updated_at="2025-06-01T12:00:00Z",
            due_at="2025-02-01T23:59:00Z",
        )
        doc = self.connector._convert_assignment_to_document(assignment)

        assert doc.id == "canvas-assignment-1-20"
        assert doc.source == DocumentSource.CANVAS
        assert doc.semantic_identifier == "Homework 1"
        assert "Due: 2025-02-01T23:59:00Z" in doc.sections[0].text

    def test_convert_assignment_without_description(self) -> None:
        from onyx.connectors.canvas.connector import CanvasAssignment

        assignment = CanvasAssignment(
            id=21,
            name="Quiz 1",
            description=None,
            html_url=f"{FAKE_BASE_URL}/courses/1/assignments/21",
            course_id=1,
            created_at="2025-01-20T00:00:00Z",
            updated_at="2025-06-01T12:00:00Z",
            due_at=None,
        )
        doc = self.connector._convert_assignment_to_document(assignment)

        section_text = doc.sections[0].text
        assert "Quiz 1" in section_text
        assert "Due:" not in section_text

    def test_convert_announcement_to_document(self) -> None:
        from onyx.connectors.canvas.connector import CanvasAnnouncement

        announcement = CanvasAnnouncement(
            id=30,
            title="Class Cancelled",
            message="<p>No class today</p>",
            html_url=f"{FAKE_BASE_URL}/courses/1/discussion_topics/30",
            posted_at="2025-06-01T12:00:00Z",
            course_id=1,
        )
        doc = self.connector._convert_announcement_to_document(announcement)

        assert doc.id == "canvas-announcement-1-30"
        assert doc.source == DocumentSource.CANVAS
        assert doc.semantic_identifier == "Class Cancelled"
        assert doc.doc_updated_at == datetime(2025, 6, 1, 12, 0, tzinfo=timezone.utc)

    def test_convert_announcement_without_posted_at(self) -> None:
        from onyx.connectors.canvas.connector import CanvasAnnouncement

        announcement = CanvasAnnouncement(
            id=31,
            title="TBD Announcement",
            message=None,
            html_url=f"{FAKE_BASE_URL}/courses/1/discussion_topics/31",
            posted_at=None,
            course_id=1,
        )
        doc = self.connector._convert_announcement_to_document(announcement)

        assert doc.doc_updated_at is None


# ---------------------------------------------------------------------------
# CanvasConnector — validate_connector_settings
# ---------------------------------------------------------------------------


class TestValidateConnectorSettings:
    @patch("onyx.connectors.canvas.connector.rl_requests")
    def test_validate_success(self, mock_requests: MagicMock) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = [_mock_course()]
        mock_requests.get.return_value = mock_response

        connector = _build_connector()
        connector.validate_connector_settings()

    @patch("onyx.connectors.canvas.connector.rl_requests")
    def test_validate_expired_credential(self, mock_requests: MagicMock) -> None:
        # First call succeeds (load_credentials), second fails (validate)
        success_response = MagicMock()
        success_response.status_code = 200
        success_response.json.return_value = [_mock_course()]

        fail_response = MagicMock()
        fail_response.status_code = 401
        fail_response.reason = "Unauthorized"
        fail_response.json.return_value = {}

        mock_requests.get.side_effect = [success_response, fail_response]

        connector = CanvasConnector(canvas_base_url=FAKE_BASE_URL)
        connector.load_credentials({"canvas_access_token": FAKE_TOKEN})

        with pytest.raises(CredentialExpiredError):
            connector.validate_connector_settings()

    @patch("onyx.connectors.canvas.connector.rl_requests")
    def test_validate_rate_limited(self, mock_requests: MagicMock) -> None:
        success_response = MagicMock()
        success_response.status_code = 200
        success_response.json.return_value = [_mock_course()]

        fail_response = MagicMock()
        fail_response.status_code = 429
        fail_response.reason = "Too Many Requests"
        fail_response.json.return_value = {}

        mock_requests.get.side_effect = [success_response, fail_response]

        connector = CanvasConnector(canvas_base_url=FAKE_BASE_URL)
        connector.load_credentials({"canvas_access_token": FAKE_TOKEN})

        with pytest.raises(ConnectorValidationError):
            connector.validate_connector_settings()

    @patch("onyx.connectors.canvas.connector.rl_requests")
    def test_validate_unexpected_error(self, mock_requests: MagicMock) -> None:
        success_response = MagicMock()
        success_response.status_code = 200
        success_response.json.return_value = [_mock_course()]

        fail_response = MagicMock()
        fail_response.status_code = 500
        fail_response.reason = "Internal Server Error"
        fail_response.json.return_value = {}

        mock_requests.get.side_effect = [success_response, fail_response]

        connector = CanvasConnector(canvas_base_url=FAKE_BASE_URL)
        connector.load_credentials({"canvas_access_token": FAKE_TOKEN})

        with pytest.raises(UnexpectedValidationError):
            connector.validate_connector_settings()


# ---------------------------------------------------------------------------
# CanvasConnector — load_from_state
# ---------------------------------------------------------------------------


class TestLoadFromState:
    @patch("onyx.connectors.canvas.connector.rl_requests")
    def test_load_from_state_returns_all_docs(self, mock_requests: MagicMock) -> None:
        course = _mock_course()
        page = _mock_page()
        assignment = _mock_assignment()
        announcement = _mock_announcement()

        def mock_get(url: str, **kwargs: object) -> MagicMock:
            resp = MagicMock()
            resp.status_code = 200
            if "/courses" in url and "/pages" in url:
                resp.json.return_value = [page]
            elif "/assignments" in url:
                resp.json.return_value = [assignment]
            elif "/announcements" in url:
                resp.json.return_value = [announcement]
            elif "/courses" in url:
                resp.json.return_value = [course]
            else:
                resp.json.return_value = []
            return resp

        mock_requests.get.side_effect = mock_get

        connector = _build_connector()
        batches = list(connector.load_from_state())

        all_docs = [doc for batch in batches for doc in batch]
        assert len(all_docs) == 3

        doc_ids = {doc.id for doc in all_docs}
        assert "canvas-page-1-10" in doc_ids
        assert "canvas-assignment-1-20" in doc_ids
        assert "canvas-announcement-1-30" in doc_ids

    @patch("onyx.connectors.canvas.connector.rl_requests")
    def test_load_from_state_continues_on_page_error(
        self, mock_requests: MagicMock
    ) -> None:
        """If pages fail for a course, assignments and announcements still load."""
        course = _mock_course()
        assignment = _mock_assignment()
        announcement = _mock_announcement()

        call_count = 0

        def mock_get(url: str, **kwargs: object) -> MagicMock:
            nonlocal call_count
            call_count += 1
            resp = MagicMock()
            resp.status_code = 200
            if "/courses" in url and "/pages" in url:
                # Simulate page fetch failure
                resp.status_code = 500
                resp.reason = "Internal Server Error"
                resp.json.return_value = {}
                return resp
            elif "/assignments" in url:
                resp.json.return_value = [assignment]
            elif "/announcements" in url:
                resp.json.return_value = [announcement]
            elif "/courses" in url:
                resp.json.return_value = [course]
            else:
                resp.json.return_value = []
            return resp

        mock_requests.get.side_effect = mock_get

        connector = _build_connector()
        batches = list(connector.load_from_state())

        all_docs = [doc for batch in batches for doc in batch]
        # Page failed but assignments + announcements should still be there
        assert len(all_docs) == 2


# ---------------------------------------------------------------------------
# CanvasConnector — poll_source
# ---------------------------------------------------------------------------


class TestPollSource:
    @patch("onyx.connectors.canvas.connector.rl_requests")
    def test_poll_filters_by_timestamp(self, mock_requests: MagicMock) -> None:
        course = _mock_course()
        # One page within window, one outside
        old_page = _mock_page(page_id=10, updated_at="2025-01-01T00:00:00Z")
        new_page = _mock_page(page_id=11, title="New Page", updated_at="2025-06-15T12:00:00Z")
        assignment = _mock_assignment(updated_at="2025-06-15T12:00:00Z")
        announcement = _mock_announcement(posted_at="2025-01-01T00:00:00Z")

        def mock_get(url: str, **kwargs: object) -> MagicMock:
            resp = MagicMock()
            resp.status_code = 200
            if "/courses" in url and "/pages" in url:
                resp.json.return_value = [old_page, new_page]
            elif "/assignments" in url:
                resp.json.return_value = [assignment]
            elif "/announcements" in url:
                resp.json.return_value = [announcement]
            elif "/courses" in url:
                resp.json.return_value = [course]
            else:
                resp.json.return_value = []
            return resp

        mock_requests.get.side_effect = mock_get

        connector = _build_connector()

        # Window: June 1 to June 30, 2025
        start = datetime(2025, 6, 1, 0, 0, tzinfo=timezone.utc).timestamp()
        end = datetime(2025, 6, 30, 0, 0, tzinfo=timezone.utc).timestamp()

        batches = list(connector.poll_source(start, end))
        all_docs = [doc for batch in batches for doc in batch]

        doc_ids = {doc.id for doc in all_docs}
        # new_page and assignment are in window; old_page and announcement are not
        assert "canvas-page-1-11" in doc_ids
        assert "canvas-assignment-1-20" in doc_ids
        assert "canvas-page-1-10" not in doc_ids
        assert "canvas-announcement-1-30" not in doc_ids

    @patch("onyx.connectors.canvas.connector.rl_requests")
    def test_poll_skips_announcement_without_posted_at(
        self, mock_requests: MagicMock
    ) -> None:
        course = _mock_course()
        announcement_no_date = _mock_announcement()
        announcement_no_date["posted_at"] = None

        def mock_get(url: str, **kwargs: object) -> MagicMock:
            resp = MagicMock()
            resp.status_code = 200
            if "/courses" in url and "/pages" in url:
                resp.json.return_value = []
            elif "/assignments" in url:
                resp.json.return_value = []
            elif "/announcements" in url:
                resp.json.return_value = [announcement_no_date]
            elif "/courses" in url:
                resp.json.return_value = [course]
            else:
                resp.json.return_value = []
            return resp

        mock_requests.get.side_effect = mock_get

        connector = _build_connector()
        start = datetime(2025, 1, 1, 0, 0, tzinfo=timezone.utc).timestamp()
        end = datetime(2025, 12, 31, 0, 0, tzinfo=timezone.utc).timestamp()

        batches = list(connector.poll_source(start, end))
        all_docs = [doc for batch in batches for doc in batch]

        assert len(all_docs) == 0
