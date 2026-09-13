from typing import Optional, List
from fastapi import APIRouter, Depends, Query, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import desc

from app.db.database import get_db
from app.db.schema import AuditLog
from app.schemas.document import VerificationResponse, PaginatedAuditLogResponse

router = APIRouter(prefix="/api/audit", tags=["audit"])

@router.get("", response_model=PaginatedAuditLogResponse)
def get_audit_trail(
    page: int = Query(1, ge=1),
    limit: int = Query(10, ge=1, le=100),
    document_type: Optional[str] = Query(None),
    prediction: Optional[str] = Query(None),
    risk_tier: Optional[str] = Query(None),
    db: Session = Depends(get_db)
):
    """
    Returns paginated audit trail of historical document screenings.
    """
    query = db.query(AuditLog)

    if document_type:
        query = query.filter(AuditLog.document_type == document_type)
    if prediction:
        query = query.filter(AuditLog.prediction == prediction)
    if risk_tier:
        if risk_tier == "high":
            query = query.filter(AuditLog.risk_score > 70)
        elif risk_tier == "medium":
            query = query.filter(AuditLog.risk_score >= 30, AuditLog.risk_score <= 70)
        elif risk_tier == "low":
            query = query.filter(AuditLog.risk_score < 30)

    total = query.count()
    items = query.order_by(desc(AuditLog.created_at)).offset((page - 1) * limit).limit(limit).all()

    return {
        "total": total,
        "page": page,
        "limit": limit,
        "items": items
    }

@router.get("/{audit_id}", response_model=VerificationResponse)
def get_audit_record(audit_id: int, db: Session = Depends(get_db)):
    """
    Returns single detailed audit record by ID.
    """
    record = db.query(AuditLog).filter(AuditLog.id == audit_id).first()
    if not record:
        raise HTTPException(status_code=404, detail="Audit log entry not found.")
    return record
