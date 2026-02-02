import json
from datetime import datetime
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.tools import tool
from core.config import Config

@tool
def save_profile_to_db(student_id: int, interest_ids: list[int], skill_ids: list[int]) -> str:
    """Save the user's profile (tags) to the database."""
    conn = Config.get_db_connection()
    try:
        cursor = conn.cursor()
        # Verify student exists
        cursor.execute("SELECT id FROM student WHERE id = %s", (student_id,))
        if not cursor.fetchone():
            # Fallback for demo
            cursor.execute("SELECT id FROM student LIMIT 1")
            res = cursor.fetchone()
            if res:
                student_id = res[0]
            else:
                return json.dumps({"status": "failed", "message": "No student found"})

        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        inserted_count = 0

        # Save Interests
        if interest_ids:
            format_strings = ','.join(['%s'] * len(interest_ids))
            cursor.execute(f"SELECT id FROM tag_1 WHERE id IN ({format_strings})", tuple(interest_ids))
            valid_ids = [row[0] for row in cursor.fetchall()]
            if valid_ids:
                data = [(student_id, tid, now) for tid in valid_ids]
                cursor.executemany(
                    "INSERT INTO tag1_stu_match (student_id, tag1_id, created_at) VALUES (%s, %s, %s) ON DUPLICATE KEY UPDATE created_at = VALUES(created_at)",
                    data
                )
                inserted_count += cursor.rowcount

        # Save Skills
        if skill_ids:
            format_strings = ','.join(['%s'] * len(skill_ids))
            cursor.execute(f"SELECT id FROM tag_2 WHERE id IN ({format_strings})", tuple(skill_ids))
            valid_ids = [row[0] for row in cursor.fetchall()]
            if valid_ids:
                data = [(student_id, tid, now) for tid in valid_ids]
                cursor.executemany(
                    "INSERT INTO tag2_stu_match (student_id, tag2_id, created_at) VALUES (%s, %s, %s) ON DUPLICATE KEY UPDATE created_at = VALUES(created_at)",
                    data
                )
                inserted_count += cursor.rowcount

        conn.commit()
        return json.dumps({"status": "success", "message": f"Updated {inserted_count} records"})
    except Exception as e:
        return json.dumps({"status": "failed", "message": str(e)})
    finally:
        conn.close()

@tool
def get_candidate_projects(interest_ids: list[int], skill_ids: list[int]) -> str:
    """Fetch potential projects from database based on tag IDs."""
    if not interest_ids and not skill_ids:
        return "[]"
        
    conn = Config.get_db_connection()
    projects = {}
    try:
        cursor = conn.cursor()
        
        # Projects matching Interests
        if interest_ids:
            format_strings = ','.join(['%s'] * len(interest_ids))
            sql = f"""
            SELECT DISTINCT pr.id, pr.title, pr.status, prt.tag1_id, t1.value
            FROM project_requirement pr
            JOIN project_requirement_tag1 prt ON pr.id = prt.requirement_id
            JOIN tag_1 t1 ON prt.tag1_id = t1.id
            WHERE prt.tag1_id IN ({format_strings}) 
            AND pr.status IN ('in_progress', 'completed', 'paused')
            LIMIT 20
            """
            cursor.execute(sql, tuple(interest_ids))
            for r in cursor.fetchall():
                pid = r[0]
                if pid not in projects:
                    projects[pid] = {"id": pid, "title": r[1], "status": r[2], "matched_tags": []}
                projects[pid]["matched_tags"].append(f"{r[4]}(Interest)")

        # Projects matching Skills
        if skill_ids:
            format_strings = ','.join(['%s'] * len(skill_ids))
            sql = f"""
            SELECT DISTINCT pr.id, pr.title, pr.status, prt.tag2_id, t2.post
            FROM project_requirement pr
            JOIN project_requirement_tag2 prt ON pr.id = prt.requirement_id
            JOIN tag_2 t2 ON prt.tag2_id = t2.id
            WHERE prt.tag2_id IN ({format_strings}) 
            AND pr.status IN ('in_progress', 'completed', 'paused')
            LIMIT 20
            """
            cursor.execute(sql, tuple(skill_ids))
            for r in cursor.fetchall():
                pid = r[0]
                if pid not in projects:
                    projects[pid] = {"id": pid, "title": r[1], "status": r[2], "matched_tags": []}
                projects[pid]["matched_tags"].append(f"{r[4]}(Skill)")
                
    except Exception as e:
        print(f"Error fetching projects: {e}")
    finally:
        conn.close()
        
    return json.dumps(list(projects.values()), ensure_ascii=False)
