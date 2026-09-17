"""POST /api/applicants/parse-document: PDF intake for a brand-new applicant (P9.1
follow-on). Read-only, no DB session, no rate limit -- unlike /applications/underwrite
and its stream, this route makes no LLM/ML calls and writes nothing, so there is
nothing here that P7.4's rate limiting was ever meant to protect.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, UploadFile

from creditsense.agents.document_intake import parse_uploaded_pdf
from creditsense.api.schemas import ParsedDocumentFields

router = APIRouter(prefix="/api")

_MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB


@router.post("/applicants/parse-document", response_model=ParsedDocumentFields)
async def parse_document(file: UploadFile) -> ParsedDocumentFields:
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF uploads are supported.")

    contents = await file.read()
    if len(contents) > _MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="PDF is larger than the 10 MB limit.")

    result = parse_uploaded_pdf(contents)
    return ParsedDocumentFields(
        fields=result.fields,
        unresolved_fields=result.unresolved_fields,
        warnings=result.warnings,
        extraction_method=result.extraction_method,  # type: ignore[arg-type]
    )
