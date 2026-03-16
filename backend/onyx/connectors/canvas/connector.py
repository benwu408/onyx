from __future__ import annotations

import copy
import re
from datetime import datetime
from datetime import timezone
from typing import Any
from typing import cast
from typing import Literal

from pydantic import BaseModel
from retry import retry
from typing_extensions import override

from onyx.access.models import ExternalAccess
from onyx.configs.app_configs import INDEX_BATCH_SIZE
from onyx.configs.constants import DocumentSource
from onyx.connectors.canvas.access import get_course_permissions
from onyx.connectors.cross_connector_utils.rate_limit_wrapper import (
    rl_requests,
)
from onyx.connectors.exceptions import ConnectorValidationError
from onyx.connectors.exceptions import CredentialExpiredError
from onyx.connectors.exceptions import UnexpectedValidationError
from onyx.connectors.interfaces import CheckpointedConnectorWithPermSync
from onyx.connectors.interfaces import CheckpointOutput
from onyx.connectors.interfaces import GenerateSlimDocumentOutput
from onyx.connectors.interfaces import SecondsSinceUnixEpoch
from onyx.connectors.interfaces import SlimConnectorWithPermSync
from onyx.connectors.models import ConnectorCheckpoint
from onyx.connectors.models import ConnectorFailure
from onyx.connectors.models import ConnectorMissingCredentialError
from onyx.connectors.models import Document
from onyx.connectors.models import DocumentFailure
from onyx.connectors.models import HierarchyNode
from onyx.connectors.models import ImageSection
from onyx.connectors.models import SlimDocument
from onyx.connectors.models import TextSection
from onyx.indexing.indexing_heartbeat import IndexingHeartbeatInterface
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
    course_id: int


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
    course_id: int


class CanvasApiClient:
    def __init__(
        self,
        bearer_token: str,
        canvas_base_url: str,
    ) -> None:
        self.bearer_token = bearer_token
        self.base_url = canvas_base_url.rstrip("/") + _CANVAS_API_VERSION

    def get(
        self,
        endpoint: str,
        params: dict[str, Any] | None = None,
        full_url: str | None = None,
    ) -> tuple[Any, str | None]:
        """Make a GET request to the Canvas API.

        Returns a tuple of (json_body, next_url).
        next_url is parsed from the Link header and is None if there are no more pages.
        If full_url is provided, it is used directly (for following pagination links).
        """
        url = full_url if full_url else self._build_url(endpoint)
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

        next_url = self._parse_next_link(response.headers.get("Link", ""))
        return json, next_url

    @staticmethod
    def _parse_next_link(link_header: str) -> str | None:
        """Extract the 'next' URL from a Canvas Link header."""
        for match in re.finditer(r'<([^>]+)>;\s*rel="next"', link_header):
            return match.group(1)
        return None

    def _build_headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.bearer_token}"}

    def _build_url(self, endpoint: str) -> str:
        return f"{self.base_url}/{endpoint.lstrip('/')}"


CanvasStage = Literal["pages", "assignments", "announcements"]


class CanvasConnectorCheckpoint(ConnectorCheckpoint):
    """Checkpoint state for resumable Canvas indexing.

    Fields:
        course_ids: Materialized list of course IDs to process.
        current_course_index: Index into course_ids for current course.
        stage: Which item type we're processing for the current course.
    """

    course_ids: list[int] = []
    current_course_index: int = 0
    stage: CanvasStage = "pages"


class CanvasConnector(
    CheckpointedConnectorWithPermSync[CanvasConnectorCheckpoint],
    SlimConnectorWithPermSync,
):
    def __init__(
        self,
        canvas_base_url: str,
        batch_size: int = INDEX_BATCH_SIZE,
    ) -> None:
        self.canvas_base_url = canvas_base_url
        self.batch_size = batch_size
        self._canvas_client: CanvasApiClient | None = None
        self._course_permissions_cache: dict[int, ExternalAccess | None] = {}

    @property
    def canvas_client(self) -> CanvasApiClient:
        if self._canvas_client is None:
            raise ConnectorMissingCredentialError(
                "Canvas API client has not been initialized. Missing credentials?"
            )
        return self._canvas_client

    def _get_course_permissions(self, course_id: int) -> ExternalAccess | None:
        """Get course permissions with caching."""
        if course_id not in self._course_permissions_cache:
            self._course_permissions_cache[course_id] = get_course_permissions(
                canvas_client=self.canvas_client,
                course_id=course_id,
            )
        return self._course_permissions_cache[course_id]

    @retry(tries=3, delay=1, backoff=2)
    def _list_courses(self) -> list[CanvasCourse]:
        """Fetch all courses accessible to the authenticated user."""
        logger.debug("Fetching Canvas courses")

        courses: list[CanvasCourse] = []
        next_url: str | None = None
        first_request = True
        while True:
            if first_request:
                response, next_url = self.canvas_client.get(
                    "courses",
                    params={
                        "per_page": "100",
                        "enrollment_state": "active",
                    },
                )
                first_request = False
            else:
                response, next_url = self.canvas_client.get(
                    "", full_url=next_url
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
            if not next_url:
                break

        return courses

    @retry(tries=3, delay=1, backoff=2)
    def _list_pages(self, course_id: int) -> list[CanvasPage]:
        """Fetch all pages for a given course."""
        logger.debug(f"Fetching pages for course {course_id}")

        pages: list[CanvasPage] = []
        next_url: str | None = None
        first_request = True
        while True:
            if first_request:
                response, next_url = self.canvas_client.get(
                    f"courses/{course_id}/pages",
                    params={
                        "per_page": "100",
                        "include[]": "body",
                    },
                )
                first_request = False
            else:
                response, next_url = self.canvas_client.get(
                    "", full_url=next_url
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
            if not next_url:
                break

        return pages

    @retry(tries=3, delay=1, backoff=2)
    def _list_assignments(self, course_id: int) -> list[CanvasAssignment]:
        """Fetch all assignments for a given course."""
        logger.debug(f"Fetching assignments for course {course_id}")

        assignments: list[CanvasAssignment] = []
        next_url: str | None = None
        first_request = True
        while True:
            if first_request:
                response, next_url = self.canvas_client.get(
                    f"courses/{course_id}/assignments",
                    params={"per_page": "100"},
                )
                first_request = False
            else:
                response, next_url = self.canvas_client.get(
                    "", full_url=next_url
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
            if not next_url:
                break

        return assignments

    @retry(tries=3, delay=1, backoff=2)
    def _list_announcements(self, course_id: int) -> list[CanvasAnnouncement]:
        """Fetch all announcements for a given course."""
        logger.debug(f"Fetching announcements for course {course_id}")

        announcements: list[CanvasAnnouncement] = []
        next_url: str | None = None
        first_request = True
        while True:
            if first_request:
                response, next_url = self.canvas_client.get(
                    "announcements",
                    params={
                        "per_page": "100",
                        "context_codes[]": f"course_{course_id}",
                    },
                )
                first_request = False
            else:
                response, next_url = self.canvas_client.get(
                    "", full_url=next_url
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
            if not next_url:
                break

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
            doc_updated_at=datetime.fromisoformat(page.updated_at).astimezone(
                timezone.utc
            ),
            metadata={"course_id": str(page.course_id)},
        )

    def _convert_assignment_to_document(
        self, assignment: CanvasAssignment
    ) -> Document:
        """Convert a Canvas assignment to a Document."""

        text_parts = [assignment.name, assignment.html_url]
        if assignment.description:
            text_parts.append(assignment.description)
        if assignment.due_at:
            text_parts.append(f"Due: {assignment.due_at}")

        sections = [
            TextSection(link=assignment.html_url, text="\n\n".join(text_parts))
        ]

        return Document(
            id=f"canvas-assignment-{assignment.course_id}-{assignment.id}",
            sections=cast(list[TextSection | ImageSection], sections),
            source=DocumentSource.CANVAS,
            semantic_identifier=assignment.name or f"Assignment {assignment.id}",
            doc_updated_at=datetime.fromisoformat(
                assignment.updated_at
            ).astimezone(timezone.utc),
            metadata={"course_id": str(assignment.course_id)},
        )

    def _convert_announcement_to_document(
        self, announcement: CanvasAnnouncement
    ) -> Document:
        """Convert a Canvas announcement to a Document."""

        text_parts = [announcement.title, announcement.html_url]
        if announcement.message:
            text_parts.append(announcement.message)

        sections = [
            TextSection(
                link=announcement.html_url, text="\n\n".join(text_parts)
            )
        ]

        doc_updated_at = None
        if announcement.posted_at:
            doc_updated_at = datetime.fromisoformat(
                announcement.posted_at
            ).astimezone(timezone.utc)

        return Document(
            id=f"canvas-announcement-{announcement.course_id}-{announcement.id}",
            sections=cast(list[TextSection | ImageSection], sections),
            source=DocumentSource.CANVAS,
            semantic_identifier=announcement.title
            or f"Announcement {announcement.id}",
            doc_updated_at=doc_updated_at,
            metadata={"course_id": str(announcement.course_id)},
        )

    def load_credentials(
        self, credentials: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Load and validate Canvas credentials."""
        self._canvas_client = CanvasApiClient(
            bearer_token=credentials["canvas_access_token"],
            canvas_base_url=self.canvas_base_url,
        )

        try:
            self._canvas_client.get("courses", params={"per_page": "1"})
        except CanvasClientRequestFailedError as e:
            if e.status_code == 401:
                raise ConnectorMissingCredentialError(
                    "Invalid Canvas API token"
                )
            raise

        return None

    def _load_from_checkpoint(
        self,
        start: SecondsSinceUnixEpoch,
        end: SecondsSinceUnixEpoch,
        checkpoint: CanvasConnectorCheckpoint,
        include_permissions: bool = False,
    ) -> CheckpointOutput[CanvasConnectorCheckpoint]:
        """Shared implementation for load_from_checkpoint and load_from_checkpoint_with_perm_sync."""
        new_checkpoint = copy.deepcopy(checkpoint)

        # First call: materialize the list of course IDs
        if not new_checkpoint.course_ids:
            courses = self._list_courses()
            new_checkpoint.course_ids = [c.id for c in courses]
            new_checkpoint.current_course_index = 0
            new_checkpoint.stage = "pages"
            logger.info(
                f"Found {len(courses)} Canvas courses to process"
            )
            new_checkpoint.has_more = len(new_checkpoint.course_ids) > 0
            return new_checkpoint

        # All courses done
        if new_checkpoint.current_course_index >= len(
            new_checkpoint.course_ids
        ):
            new_checkpoint.has_more = False
            return new_checkpoint

        course_id = new_checkpoint.course_ids[
            new_checkpoint.current_course_index
        ]
        stage = new_checkpoint.stage

        if stage not in ("pages", "assignments", "announcements"):
            raise ValueError(
                f"Invalid checkpoint stage: {stage!r}"
            )

        def _in_time_window(timestamp_str: str) -> bool:
            ts = (
                datetime.fromisoformat(timestamp_str)
                .astimezone(timezone.utc)
                .timestamp()
            )
            return start < ts <= end

        def _maybe_attach_permissions(document: Document) -> Document:
            if include_permissions:
                document.external_access = self._get_course_permissions(
                    course_id
                )
            return document

        # Process current stage for current course.
        # Only advance if the listing succeeds — on failure the stage
        # stays the same so the framework retries it next call.
        if stage == "pages":
            try:
                pages = self._list_pages(course_id)
            except Exception as e:
                logger.warning(
                    f"Failed to list pages for course {course_id}: {e}"
                )
                return new_checkpoint

            for page in pages:
                if _in_time_window(page.updated_at):
                    try:
                        doc = self._convert_page_to_document(page)
                        yield _maybe_attach_permissions(doc)
                    except Exception as e:
                        yield ConnectorFailure(
                            failed_document=DocumentFailure(
                                document_id=f"canvas-page-{course_id}-{page.page_id}",
                                document_link=f"{self.canvas_base_url}/courses/{course_id}/pages/{page.url}",
                            ),
                            failure_message=f"Failed to convert page: {e}",
                            exception=e,
                        )

            new_checkpoint.stage = "assignments"

        elif stage == "assignments":
            try:
                assignments = self._list_assignments(course_id)
            except Exception as e:
                logger.warning(
                    f"Failed to list assignments for course {course_id}: {e}"
                )
                return new_checkpoint

            for assignment in assignments:
                if _in_time_window(assignment.updated_at):
                    try:
                        doc = self._convert_assignment_to_document(assignment)
                        yield _maybe_attach_permissions(doc)
                    except Exception as e:
                        yield ConnectorFailure(
                            failed_document=DocumentFailure(
                                document_id=f"canvas-assignment-{course_id}-{assignment.id}",
                                document_link=assignment.html_url,
                            ),
                            failure_message=f"Failed to convert assignment: {e}",
                            exception=e,
                        )

            new_checkpoint.stage = "announcements"

        elif stage == "announcements":
            try:
                announcements = self._list_announcements(course_id)
            except Exception as e:
                logger.warning(
                    f"Failed to list announcements for course {course_id}: {e}"
                )
                return new_checkpoint

            for announcement in announcements:
                if not announcement.posted_at:
                    continue
                if _in_time_window(announcement.posted_at):
                    try:
                        doc = self._convert_announcement_to_document(
                            announcement
                        )
                        yield _maybe_attach_permissions(doc)
                    except Exception as e:
                        yield ConnectorFailure(
                            failed_document=DocumentFailure(
                                document_id=f"canvas-announcement-{course_id}-{announcement.id}",
                                document_link=announcement.html_url,
                            ),
                            failure_message=f"Failed to convert announcement: {e}",
                            exception=e,
                        )

            # Done with all 3 stages for this course, advance to next
            new_checkpoint.current_course_index += 1
            new_checkpoint.stage = "pages"

        new_checkpoint.has_more = new_checkpoint.current_course_index < len(
            new_checkpoint.course_ids
        )
        return new_checkpoint

    @override
    def load_from_checkpoint(
        self,
        start: SecondsSinceUnixEpoch,
        end: SecondsSinceUnixEpoch,
        checkpoint: CanvasConnectorCheckpoint,
    ) -> CheckpointOutput[CanvasConnectorCheckpoint]:
        return self._load_from_checkpoint(
            start, end, checkpoint, include_permissions=False
        )

    @override
    def load_from_checkpoint_with_perm_sync(
        self,
        start: SecondsSinceUnixEpoch,
        end: SecondsSinceUnixEpoch,
        checkpoint: CanvasConnectorCheckpoint,
    ) -> CheckpointOutput[CanvasConnectorCheckpoint]:
        """Load documents from checkpoint with permission information included."""
        return self._load_from_checkpoint(
            start, end, checkpoint, include_permissions=True
        )

    @override
    def build_dummy_checkpoint(self) -> CanvasConnectorCheckpoint:
        return CanvasConnectorCheckpoint(has_more=True)

    @override
    def validate_checkpoint_json(
        self, checkpoint_json: str
    ) -> CanvasConnectorCheckpoint:
        return CanvasConnectorCheckpoint.model_validate_json(checkpoint_json)

    @override
    def retrieve_all_slim_docs_perm_sync(
        self,
        start: SecondsSinceUnixEpoch | None = None,  # noqa: ARG002
        end: SecondsSinceUnixEpoch | None = None,  # noqa: ARG002
        callback: IndexingHeartbeatInterface | None = None,
    ) -> GenerateSlimDocumentOutput:
        """Return slim documents with permission info for all courses."""
        batch: list[SlimDocument | HierarchyNode] = []
        courses = self._list_courses()

        for course in courses:
            course_id = course.id
            permissions = self._get_course_permissions(course_id)

            # Pages
            try:
                pages = self._list_pages(course_id)
                for page in pages:
                    batch.append(
                        SlimDocument(
                            id=f"canvas-page-{course_id}-{page.page_id}",
                            external_access=permissions,
                        )
                    )
                    if len(batch) >= self.batch_size:
                        yield batch
                        batch = []
                        if callback and callback.should_stop():
                            raise RuntimeError(
                                "canvas_perm_sync: Stop signal detected"
                            )
            except RuntimeError:
                raise
            except Exception as e:
                logger.warning(
                    f"Failed to list pages for course {course_id}: {e}"
                )

            # Assignments
            try:
                assignments = self._list_assignments(course_id)
                for assignment in assignments:
                    batch.append(
                        SlimDocument(
                            id=f"canvas-assignment-{course_id}-{assignment.id}",
                            external_access=permissions,
                        )
                    )
                    if len(batch) >= self.batch_size:
                        yield batch
                        batch = []
                        if callback and callback.should_stop():
                            raise RuntimeError(
                                "canvas_perm_sync: Stop signal detected"
                            )
            except RuntimeError:
                raise
            except Exception as e:
                logger.warning(
                    f"Failed to list assignments for course {course_id}: {e}"
                )

            # Announcements
            try:
                announcements = self._list_announcements(course_id)
                for announcement in announcements:
                    batch.append(
                        SlimDocument(
                            id=f"canvas-announcement-{course_id}-{announcement.id}",
                            external_access=permissions,
                        )
                    )
                    if len(batch) >= self.batch_size:
                        yield batch
                        batch = []
                        if callback and callback.should_stop():
                            raise RuntimeError(
                                "canvas_perm_sync: Stop signal detected"
                            )
            except RuntimeError:
                raise
            except Exception as e:
                logger.warning(
                    f"Failed to list announcements for course {course_id}: {e}"
                )

            if callback:
                callback.progress("canvas_perm_sync", 1)

        if batch:
            yield batch

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
