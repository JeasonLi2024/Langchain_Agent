import json
import asyncio
from datetime import datetime
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.tools import tool
from core.config import Config


def _safe_json_loads(value, default):
    if not value:
        return default
    try:
        return json.loads(value)
    except Exception:
        return default


def _resolve_student_id(cursor, student_id: int | None, user_id: int | None) -> int | None:
    if student_id:
        cursor.execute("SELECT id FROM student WHERE id = %s", (student_id,))
        row = cursor.fetchone()
        if row:
            return row[0]

    if user_id:
        cursor.execute("SELECT id FROM student WHERE user_id = %s", (user_id,))
        row = cursor.fetchone()
        if row:
            return row[0]

    return None


def _load_student_context_sync(student_id: int | None = None, user_id: int | None = None) -> str:
    if not student_id and not user_id:
        return json.dumps({
            "status": "failed",
            "message": "student_id or user_id is required",
            "student_id": None,
        }, ensure_ascii=False)
    conn = Config.get_db_connection()
    try:
        cursor = conn.cursor()
        resolved_student_id = _resolve_student_id(cursor, student_id, user_id)
        if not resolved_student_id:
            return json.dumps({
                "status": "failed",
                "message": "Student not found",
                "student_id": None,
            }, ensure_ascii=False)

        cursor.execute(
            """
            SELECT s.id, s.user_id, s.student_id, s.school_id, u.school, s.major, s.grade,
                   s.education_level, s.status, s.verification, s.expected_graduation
            FROM student s
            LEFT JOIN university u ON s.school_id = u.id
            WHERE s.id = %s
            """,
            (resolved_student_id,),
        )
        row = cursor.fetchone()
        student_profile = {
            "id": row[0],
            "user_id": row[1],
            "student_no": row[2],
            "school_id": row[3],
            "school_name": row[4],
            "major": row[5],
            "grade": row[6],
            "education_level": row[7],
            "status": row[8],
            "verification": row[9],
            "expected_graduation": str(row[10]) if row[10] else None,
        }

        cursor.execute(
            """
            SELECT t.id, t.value
            FROM tag1_stu_match m
            JOIN tag_1 t ON m.tag1_id = t.id
            WHERE m.student_id = %s
            ORDER BY m.created_at DESC, t.id ASC
            """,
            (resolved_student_id,),
        )
        confirmed_interest_tags = [
            {"id": tag_id, "name": name}
            for tag_id, name in cursor.fetchall()
        ]

        cursor.execute(
            """
            SELECT t.id, t.post
            FROM tag2_stu_match m
            JOIN tag_2 t ON m.tag2_id = t.id
            WHERE m.student_id = %s
            ORDER BY m.created_at DESC, t.id ASC
            """,
            (resolved_student_id,),
        )
        confirmed_skill_tags = [
            {"id": tag_id, "name": name}
            for tag_id, name in cursor.fetchall()
        ]

        current_profile = {
            "profile_summary": "",
            "candidate_interest_tags": [],
            "candidate_skill_tags": [],
        }
        try:
            cursor.execute(
                """
                SELECT profile_summary, candidate_interest_tags_json, candidate_skill_tags_json, updated_at
                FROM student_profile_current
                WHERE student_id = %s
                LIMIT 1
                """,
                (resolved_student_id,),
            )
            profile_row = cursor.fetchone()
            if profile_row:
                current_profile = {
                    "profile_summary": profile_row[0] or "",
                    "candidate_interest_tags": _safe_json_loads(profile_row[1], []),
                    "candidate_skill_tags": _safe_json_loads(profile_row[2], []),
                    "updated_at": str(profile_row[3]) if profile_row[3] else None,
                }
        except Exception:
            current_profile["table_missing"] = True

        return json.dumps({
            "status": "success",
            "student_id": resolved_student_id,
            "user_id": student_profile["user_id"],
            "student_profile": student_profile,
            "confirmed_interest_tags": confirmed_interest_tags,
            "confirmed_skill_tags": confirmed_skill_tags,
            "current_profile": current_profile,
        }, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"status": "failed", "message": str(e)}, ensure_ascii=False)
    finally:
        conn.close()


@tool
async def load_student_context(student_id: int | None = None, user_id: int | None = None) -> str:
    """Load Student profile, confirmed tags, and current recommendation profile."""
    return await asyncio.to_thread(_load_student_context_sync, student_id, user_id)


def _update_student_core_fields_sync(
    student_id: int,
    school_name: str | None = None,
    major: str | None = None,
    grade: str | None = None,
    education_level: str | None = None,
) -> str:
    """Update Student core fields when the user explicitly provides them."""
    conn = Config.get_db_connection()
    try:
        cursor = conn.cursor()
        updates = []
        params = []

        if school_name:
            cursor.execute(
                "SELECT id FROM university WHERE school = %s LIMIT 1",
                (school_name.strip(),),
            )
            school_row = cursor.fetchone()
            if school_row:
                updates.append("school_id = %s")
                params.append(school_row[0])

        if major:
            updates.append("major = %s")
            params.append(major.strip())

        if grade:
            updates.append("grade = %s")
            params.append(grade.strip())

        if education_level:
            updates.append("education_level = %s")
            params.append(education_level.strip())

        if not updates:
            return json.dumps({"status": "skipped", "message": "No valid Student fields to update"}, ensure_ascii=False)

        params.append(student_id)
        cursor.execute(
            f"UPDATE student SET {', '.join(updates)}, updated_at = %s WHERE id = %s",
            tuple(params[:-1] + [datetime.now().strftime('%Y-%m-%d %H:%M:%S'), params[-1]]),
        )
        conn.commit()
        return json.dumps({"status": "success", "student_id": student_id}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"status": "failed", "message": str(e)}, ensure_ascii=False)
    finally:
        conn.close()


@tool
async def update_student_core_fields(
    student_id: int,
    school_name: str | None = None,
    major: str | None = None,
    grade: str | None = None,
    education_level: str | None = None,
) -> str:
    """Update Student core fields when the user explicitly provides them."""
    return await asyncio.to_thread(
        _update_student_core_fields_sync,
        student_id,
        school_name,
        major,
        grade,
        education_level,
    )


def _add_student_tags_sync(student_id: int, interest_ids: list[int] | None = None, skill_ids: list[int] | None = None) -> str:
    """Append confirmed tags into tag1_stu_match and tag2_stu_match."""
    conn = Config.get_db_connection()
    try:
        cursor = conn.cursor()
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        inserted_count = 0

        if interest_ids:
            format_strings = ",".join(["%s"] * len(interest_ids))
            cursor.execute(f"SELECT id FROM tag_1 WHERE id IN ({format_strings})", tuple(interest_ids))
            valid_ids = [row[0] for row in cursor.fetchall()]
            if valid_ids:
                data = [(student_id, tid, now) for tid in valid_ids]
                cursor.executemany(
                    """
                    INSERT INTO tag1_stu_match (student_id, tag1_id, created_at)
                    VALUES (%s, %s, %s)
                    ON DUPLICATE KEY UPDATE created_at = VALUES(created_at)
                    """,
                    data,
                )
                inserted_count += cursor.rowcount

        if skill_ids:
            format_strings = ",".join(["%s"] * len(skill_ids))
            cursor.execute(f"SELECT id FROM tag_2 WHERE id IN ({format_strings})", tuple(skill_ids))
            valid_ids = [row[0] for row in cursor.fetchall()]
            if valid_ids:
                data = [(student_id, tid, now) for tid in valid_ids]
                cursor.executemany(
                    """
                    INSERT INTO tag2_stu_match (student_id, tag2_id, created_at)
                    VALUES (%s, %s, %s)
                    ON DUPLICATE KEY UPDATE created_at = VALUES(created_at)
                    """,
                    data,
                )
                inserted_count += cursor.rowcount

        conn.commit()
        return json.dumps({"status": "success", "updated_count": inserted_count}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"status": "failed", "message": str(e)}, ensure_ascii=False)
    finally:
        conn.close()


@tool
async def add_student_tags(student_id: int, interest_ids: list[int] | None = None, skill_ids: list[int] | None = None) -> str:
    """Append confirmed tags into tag1_stu_match and tag2_stu_match."""
    return await asyncio.to_thread(_add_student_tags_sync, student_id, interest_ids, skill_ids)


def _upsert_student_profile_current_sync(
    student_id: int,
    profile_summary: str,
    candidate_interest_tags_json: str,
    candidate_skill_tags_json: str,
) -> str:
    """Upsert student_profile_current if the table exists."""
    conn = Config.get_db_connection()
    try:
        cursor = conn.cursor()
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        cursor.execute(
            """
            INSERT INTO student_profile_current
                (student_id, profile_summary, candidate_interest_tags_json, candidate_skill_tags_json, updated_at)
            VALUES (%s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
                profile_summary = VALUES(profile_summary),
                candidate_interest_tags_json = VALUES(candidate_interest_tags_json),
                candidate_skill_tags_json = VALUES(candidate_skill_tags_json),
                updated_at = VALUES(updated_at)
            """,
            (student_id, profile_summary, candidate_interest_tags_json, candidate_skill_tags_json, now),
        )
        conn.commit()
        return json.dumps({"status": "success", "student_id": student_id}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({
            "status": "failed",
            "message": str(e),
            "hint": "student_profile_current table may be missing",
        }, ensure_ascii=False)
    finally:
        conn.close()


@tool
async def upsert_student_profile_current(
    student_id: int,
    profile_summary: str,
    candidate_interest_tags_json: str,
    candidate_skill_tags_json: str,
) -> str:
    """Upsert student_profile_current if the table exists."""
    return await asyncio.to_thread(
        _upsert_student_profile_current_sync,
        student_id,
        profile_summary,
        candidate_interest_tags_json,
        candidate_skill_tags_json,
    )


def _save_profile_evidence_sync(
    student_id: int,
    evidence_type: str,
    evidence_text: str,
    field_mapping_json: str = "{}",
    evidence_confidence: float = 0.8,
    source: str = "agent",
    session_id: str | None = None,
) -> str:
    """Persist profile evidence if the table exists."""
    conn = Config.get_db_connection()
    try:
        cursor = conn.cursor()
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        cursor.execute(
            """
            INSERT INTO student_profile_evidence
                (student_id, evidence_type, evidence_text, field_mapping_json, evidence_confidence, source, session_id, created_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                student_id,
                evidence_type,
                evidence_text,
                field_mapping_json,
                evidence_confidence,
                source,
                session_id,
                now,
            ),
        )
        conn.commit()
        return json.dumps({"status": "success", "student_id": student_id}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({
            "status": "failed",
            "message": str(e),
            "hint": "student_profile_evidence table may be missing",
        }, ensure_ascii=False)
    finally:
        conn.close()


@tool
async def save_profile_evidence(
    student_id: int,
    evidence_type: str,
    evidence_text: str,
    field_mapping_json: str = "{}",
    evidence_confidence: float = 0.8,
    source: str = "agent",
    session_id: str | None = None,
) -> str:
    """Persist profile evidence if the table exists."""
    return await asyncio.to_thread(
        _save_profile_evidence_sync,
        student_id,
        evidence_type,
        evidence_text,
        field_mapping_json,
        evidence_confidence,
        source,
        session_id,
    )
