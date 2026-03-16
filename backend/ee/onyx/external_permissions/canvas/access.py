"""
EE implementation: fetch Canvas course permissions from the enrollments API.

Only Teacher, TA, and Designer enrollments grant access to course documents.
"""

from onyx.access.models import ExternalAccess
from onyx.connectors.canvas.connector import CanvasApiClient
from onyx.utils.logger import setup_logger

logger = setup_logger()

_TEACHER_ENROLLMENT_TYPES = [
    "TeacherEnrollment",
    "TaEnrollment",
    "DesignerEnrollment",
]


def get_course_permissions(
    canvas_client: CanvasApiClient,
    course_id: int,
) -> ExternalAccess:
    """Fetch teacher/TA/designer emails for a Canvas course.

    Calls the Canvas enrollments API filtered by role type,
    paginates via Link headers, and returns an ExternalAccess
    with those emails.

    Returns ExternalAccess.empty() on failure (safe fallback —
    documents become private rather than public).
    """
    emails: set[str] = set()

    try:
        next_url: str | None = None
        first_request = True

        while True:
            if first_request:
                response, next_url = canvas_client.get(
                    f"courses/{course_id}/enrollments",
                    params={
                        "type[]": _TEACHER_ENROLLMENT_TYPES,
                        "per_page": "100",
                        "state[]": "active",
                    },
                )
                first_request = False
            else:
                response, next_url = canvas_client.get(
                    "", full_url=next_url
                )

            if not response:
                break

            for enrollment in response:
                user = enrollment.get("user", {})
                email = user.get("login_id") or user.get("email")
                if email:
                    emails.add(email)

            if not next_url:
                break

    except Exception as e:
        logger.warning(
            f"Failed to fetch enrollments for course {course_id}: {e}. "
            "Falling back to empty access (documents will be private)."
        )
        return ExternalAccess.empty()

    if not emails:
        logger.debug(
            f"No teacher/TA/designer enrollments found for course {course_id}"
        )
        return ExternalAccess.empty()

    return ExternalAccess(
        external_user_emails=emails,
        external_user_group_ids=set(),
        is_public=False,
    )
