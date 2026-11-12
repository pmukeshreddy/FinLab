from pydantic import BaseModel

from finpipe.storage.models import RejectionReason


class JudgeResult(BaseModel):
    paper_id: str
    scope_id: str
    accepted: bool
    rejection_reason: RejectionReason | None
    reasoning: str
