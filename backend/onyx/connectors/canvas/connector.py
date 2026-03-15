from __future__ import annotations

import copy
from datetime import datetime
from datetime import timezone
from typing import Any
from typing import cast
from typing import Optional

from pydantic import BaseModel
from retry import retry
from typing_extensions import override

from onyx.configs.app_configs import INDEX_BATCH_SIZE
from onyx.configs.constants import DocumentSource
from onyx.connectors.cross_connector_utils.rate_limit_wrapper import (
    rl_requests,
)
from onyx.connectors.exceptions import ConnectorValidationError
from onyx.connectors.exceptions import CredentialExpiredError
from onyx.connectors.exceptions import UnexpectedValidationError
from onyx.connectors.interfaces import CheckpointedConnector
from onyx.connectors.interfaces import CheckpointOutput
from onyx.connectors.interfaces import SecondsSinceUnixEpoch
from onyx.connectors.models import ConnectorCheckpoint
from onyx.connectors.models import ConnectorFailure
from onyx.connectors.models import ConnectorMissingCredentialError
from onyx.connectors.models import Document
from onyx.connectors.models import DocumentFailure
from onyx.connectors.models import ImageSection
from onyx.connectors.models import TextSection
from onyx.utils.logger import setup_logger

_CANVAS_CALL_TIMEOUT = 30
_CANVAS_API_VERSION = "/api/v1"

logger = setup_logger()

class CanvasClientRequestFailedError(ConnectionError):
    def __init__(self, message: str, status_code: int):
        super().__init__(
            f"Canvas API request failed with status {status_code}: {message}"
        )
        self.status_code = status_code


class CanvasCourse(BaseModel):
    id: int
    name: str
    course_code: str
    created_at: str
    workflow_state: str


class CanvasPage(BaseModel):
    page_id: int
    url: str
    title: str
    body: str | None
    created_at: str
    updated_at: str
    course_id: int      # you track this, not provided by canvas but useful for your processing


class CanvasAssignment(BaseModel):
    id: int
    name: str
    description: str | None
    html_url: str
    course_id: int
    created_at: str
    updated_at: str
    due_at: str | None


class CanvasAnnouncement(BaseModel):
    id: int
    title: str
    message: str | None
    html_url: str
    posted_at: str | None
    course_id: int           # you track this


class CanvasApiClient:
    def __init__(
        self,
        bearer_token: str,
        canvas_base_url: str,
    ) -> None:
        self.bearer_token = bearer_token
        self.base_url = canvas_base_url.rstrip("/") + _CANVAS_API_VERSION

    def get(
        self, endpoint: str, params: Optional[dict[str, str]] = None
    ) -> Any:
        url = self._build_url(endpoint)
        headers = self._build_headers()

        response = rl_requests.get(
            url, headers=headers, params=params, timeout=_CANVAS_CALL_TIMEOUT
        )

        try:
            json = response.json()
        except Exception:
            json = {}

        if response.status_code >= 300:
            error = response.reason
            response_error = json.get("error", {}).get("message", "")
            if response_error:
                error = response_error
            raise CanvasClientRequestFailedError(error, response.status_code)
        
        return json
    
    def _build_headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.bearer_token}"}
    
    def _build_url(self, endpoint: str) -> str:
        return f"{self.base_url}/{endpoint.lstrip('/')}"
    
class CanvasConnector(LoadConnector, PollConnector):
    def __init__(
        self,
        canvas_base_url: str,
        batch_size: int = INDEX_BATCH_SIZE,
    ) -> None:
        self.canvas_base_url = canvas_base_url
        self.batch_size = batch_size
        self._canvas_client: CanvasApiClient | None = None

    @property
    def canvas_client(self) -> CanvasApiClient:
        if self._canvas_client is None:
            raise ConnectorMissingCredentialError("Canvas API client has not been initialized. Missing credentials?")
        return self._canvas_client 
    
    @retry(tries=3, delay=1, backoff=2)
    def _list_courses(self) -> list[CanvasCourse]:
        """Fetch all courses accessible to the authenticated user."""
        logger.debug("Fetching Canvas courses")

        courses: list[CanvasCourse] = []
        page = 1
        while True:
            response = self.canvas_client.get(
                "courses",
                params={
                    "per_page": "100",
                    "page": str(page),
                    "enrollment_state": "active",
                },
            )
            if not response:
                break
            courses.extend(
                CanvasCourse(
                    id=course["id"],
                    name=course["name"],
                    course_code=course["course_code"],
                    created_at=course["created_at"],
                    workflow_state=course["workflow_state"],
                )
                for course in response
            )
            if len(response) < 100:
                break
            page += 1

        return courses
    
    @retry(tries=3, delay=1, backoff=2)
    def _list_pages(self, course_id: int) -> list[CanvasPage]:
        """Fetch all pages for a given course."""
        logger.debug(f"Fetching pages for course {course_id}")

        pages: list[CanvasPage] = []
        page_num = 1
        while True:
            response = self.canvas_client.get(
                f"courses/{course_id}/pages",
                params={
                    "per_page": "100",
                    "page": str(page_num),
                    "include[]": "body",
                },
            )
            if not response:
                break
            pages.extend(
                CanvasPage(
                    page_id=p["page_id"],
                    url=p["url"],
                    title=p["title"],
                    body=p.get("body"),
                    created_at=p["created_at"],
                    updated_at=p["updated_at"],
                    course_id=course_id,
                )
                for p in response
            )
            if len(response) < 100:
                break
            page_num += 1

        return pages
    
    @retry(tries=3, delay=1, backoff=2)
    def _list_assignments(self, course_id: int) -> list[CanvasAssignment]:
        """Fetch all assignments for a given course."""
        logger.debug(f"Fetching assignments for course {course_id}")

        assignments: list[CanvasAssignment] = []
        page = 1
        while True:
            response = self.canvas_client.get(
                f"courses/{course_id}/assignments",
                params={"per_page": "100", "page": str(page)},
            )
            if not response:
                break
            assignments.extend(
                CanvasAssignment(
                    id=assignment["id"],
                    name=assignment["name"],
                    description=assignment.get("description"),
                    html_url=assignment["html_url"],
                    course_id=course_id,
                    created_at=assignment["created_at"],
                    updated_at=assignment["updated_at"],
                    due_at=assignment.get("due_at"),
                )
                for assignment in response
            )
            if len(response) < 100:
                break
            page += 1

        return assignments
    
    @retry(tries=3, delay=1, backoff=2)
    def _list_announcements(self, course_id: int) -> list[CanvasAnnouncement]:
        """Fetch all announcements for a given course."""
        logger.debug(f"Fetching announcements for course {course_id}")

        announcements: list[CanvasAnnouncement] = []
        page = 1
        while True:
            response = self.canvas_client.get(
                "announcements",
                params={
                    "per_page": "100",
                    "page": str(page),
                    "context_codes[]": f"course_{course_id}",
                },
            )
            if not response:
                break
            announcements.extend(
                CanvasAnnouncement(
                    id=a["id"],
                    title=a["title"],
                    message=a.get("message"),
                    html_url=a["html_url"],
                    posted_at=a.get("posted_at"),
                    course_id=course_id,
                )
                for a in response
            )
            if len(response) < 100:
                break
            page += 1

        return announcements
    
    def _convert_page_to_document(self, page: CanvasPage) -> Document:
        """Convert a Canvas page to a Document."""
        
        link = f"{self.canvas_base_url}/courses/{page.course_id}/pages/{page.url}"

        text_parts = [page.title, link]
        if page.body:
            text_parts.append(page.body)

        sections = [TextSection(link=link, text="\n\n".join(text_parts))]

        return Document(
            id=f"canvas-page-{page.course_id}-{page.page_id}",
            sections=cast(list[TextSection | ImageSection], sections),
            source=DocumentSource.CANVAS,
            semantic_identifier=page.title or f"Page {page.page_id}",
            doc_updated_at=datetime.fromisoformat(page.updated_at).astimezone(timezone.utc),
            metadata={"course_id": str(page.course_id)},
        )
    
    def _convert_assignment_to_document(self, assignment: CanvasAssignment) -> Document:
        """Convert a Canvas assignment to a Document."""

        text_parts = [assignment.name, assignment.html_url]
        if assignment.description:
            text_parts.append(assignment.description)
        if assignment.due_at:
            text_parts.append(f"Due: {assignment.due_at}")

        sections = [TextSection(link=assignment.html_url, text="\n\n".join(text_parts))]

        return Document(
            id=f"canvas-assignment-{assignment.course_id}-{assignment.id}",
            sections=cast(list[TextSection | ImageSection], sections),
            source=DocumentSource.CANVAS,
            semantic_identifier=assignment.name or f"Assignment {assignment.id}",
            doc_updated_at=datetime.fromisoformat(assignment.updated_at).astimezone(timezone.utc),
            metadata={"course_id": str(assignment.course_id)},
        )

    def _convert_announcement_to_document(self, announcement: CanvasAnnouncement) -> Document:
        """Convert a Canvas announcement to a Document."""

        text_parts = [announcement.title, announcement.html_url]
        if announcement.message:
            text_parts.append(announcement.message)

        sections = [TextSection(link=announcement.html_url, text="\n\n".join(text_parts))]

        doc_updated_at = None
        if announcement.posted_at:
            doc_updated_at = datetime.fromisoformat(announcement.posted_at).astimezone(timezone.utc)

        return Document(
            id=f"canvas-announcement-{announcement.course_id}-{announcement.id}",
            sections=cast(list[TextSection | ImageSection], sections),
            source=DocumentSource.CANVAS,
            semantic_identifier=announcement.title or f"Announcement {announcement.id}",
            doc_updated_at=doc_updated_at,
            metadata={"course_id": str(announcement.course_id)},
        )

    def load_credentials(self, credentials: dict[str, Any]) -> dict[str, Any] | None:
        """Load and validate Canvas credentials."""
        self._canvas_client = CanvasApiClient(
            bearer_token=credentials["canvas_access_token"],
            canvas_base_url=self.canvas_base_url,
        )

        try:
            self._canvas_client.get("courses", params={"per_page": "1"})
        except CanvasClientRequestFailedError as e:
            if e.status_code == 401:
                raise ConnectorMissingCredentialError("Invalid Canvas API token")
            raise

        return None
    
    def load_from_state(self) -> GenerateDocumentsOutput:
        """Load all documents from Canvas."""

        def _iter_documents() -> Generator[Document, None, None]:
            courses = self._list_courses()
            logger.info(f"Found {len(courses)} Canvas courses to process")

            for course in courses:
                logger.debug(f"Processing course: {course.name} ({course.id})")

                try:
                    pages = self._list_pages(course.id)
                    for page in pages:
                        yield self._convert_page_to_document(page)
                except Exception as e:
                    logger.warning(f"Failed to list pages for course {course.id}: {e}")

                try:
                    assignments = self._list_assignments(course.id)
                    for assignment in assignments:
                        yield self._convert_assignment_to_document(assignment)
                except Exception as e:
                    logger.warning(f"Failed to list assignments for course {course.id}: {e}")

                try:
                    announcements = self._list_announcements(course.id)
                    for announcement in announcements:
                        yield self._convert_announcement_to_document(announcement)
                except Exception as e:
                    logger.warning(f"Failed to list announcements for course {course.id}: {e}")

        return batch_generator(_iter_documents(), self.batch_size)

    def poll_source(
        self, start: SecondsSinceUnixEpoch, end: SecondsSinceUnixEpoch
    ) -> GenerateDocumentsOutput:
        """Poll Canvas for documents updated between start and end timestamps."""

        def _iter_documents() -> Generator[Document, None, None]:
            courses = self._list_courses()
            logger.info(
                f"Polling {len(courses)} Canvas courses for updates between {start} and {end}"
            )

            for course in courses:
                try:
                    pages = self._list_pages(course.id)
                    for page in pages:
                        page_timestamp = (
                            datetime.fromisoformat(page.updated_at)
                            .astimezone(timezone.utc)
                            .timestamp()
                        )
                        if start < page_timestamp <= end:
                            yield self._convert_page_to_document(page)
                except Exception as e:
                    logger.warning(f"Failed to poll pages for course {course.id}: {e}")

                try:
                    assignments = self._list_assignments(course.id)
                    for assignment in assignments:
                        assignment_timestamp = (
                            datetime.fromisoformat(assignment.updated_at)
                            .astimezone(timezone.utc)
                            .timestamp()
                        )
                        if start < assignment_timestamp <= end:
                            yield self._convert_assignment_to_document(assignment)
                except Exception as e:
                    logger.warning(f"Failed to poll assignments for course {course.id}: {e}")

                try:
                    announcements = self._list_announcements(course.id)
                    for announcement in announcements:
                        if not announcement.posted_at:
                            continue
                        announcement_timestamp = (
                            datetime.fromisoformat(announcement.posted_at)
                            .astimezone(timezone.utc)
                            .timestamp()
                        )
                        if start < announcement_timestamp <= end:
                            yield self._convert_announcement_to_document(announcement)
                except Exception as e:
                    logger.warning(f"Failed to poll announcements for course {course.id}: {e}")

        return batch_generator(_iter_documents(), self.batch_size)

    def validate_connector_settings(self) -> None:
        """Validate Canvas connector settings by testing API access."""
        try:
            self.canvas_client.get("courses", params={"per_page": "1"})
            logger.info("Canvas connector settings validated successfully")
        except CanvasClientRequestFailedError as e:
            if e.status_code == 401:
                raise CredentialExpiredError(
                    "Canvas credential appears to be invalid or expired (HTTP 401)."
                )
            elif e.status_code == 429:
                raise ConnectorValidationError(
                    "Validation failed due to Canvas rate-limits being exceeded (HTTP 429). "
                    "Please try again later."
                )
            else:
                raise UnexpectedValidationError(
                    f"Unexpected Canvas HTTP error (status={e.status_code}): {e}"
                )
        except Exception as exc:
            raise UnexpectedValidationError(
                f"Unexpected error during Canvas settings validation: {exc}"
            )
