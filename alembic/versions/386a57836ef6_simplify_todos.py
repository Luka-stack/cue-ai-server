"""simplify_todos

Revision ID: 386a57836ef6
Revises: d4e5f6a7b8c9
Create Date: 2026-10-11 17:39:00.112153

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "386a57836ef6"
down_revision: Union[str, Sequence[str], None] = "d4e5f6a7b8c9"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. Rescue solutions: append them to content as a bullet list.
    op.execute("""
        UPDATE items
            SET content = CONCAT_WS(E'\\n\\n', NULLIF(content, ''), 
                'Solutions:' || E'\\n' || (
                    SELECT string_agg('- ' || s, E'\\n')
                        FROM jsonb_array_elements_text(solutions) AS s))
        WHERE type = 'todo' AND jsonb_array_length(solutions) > 0
    """)

    # 2. Drop constraints
    op.drop_constraint(
        op.f("ck_items_todo_status_matches_completed"), "items", type_="check"
    )
    op.drop_constraint(op.f("ck_items_status_valid"), "items", type_="check")
    op.drop_constraint(op.f("ck_items_note_has_no_todo_fields"), "items", type_="check")

    # 3. Drop columns
    op.drop_column("items", "solutions")
    op.drop_column("items", "status")

    # 4. Rename date
    op.alter_column("items", "date", new_column_name="deadline")
    op.execute(
        "ALTER INDEX ix_items_user_id_type_date RENAME TO ix_items_user_id_type_deadline"
    )

    # 5. New rules.
    op.create_check_constraint(
        op.f("ck_items_note_has_no_todo_fields"),
        "items",
        "type <> 'note' OR (deadline IS NULL AND completed IS NULL AND time_to_complete IS NULL)",
    )
    op.create_check_constraint(
        op.f("ck_items_todo_has_completed"),
        "items",
        "type <> 'todo' OR completed IS NOT NULL",
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_constraint(op.f("ck_items_note_has_no_todo_fields"), "items", type_="check")
    op.drop_constraint(op.f("ck_items_todo_has_completed"), "items", type_="check")

    op.execute(
        "ALTER INDEX ix_items_user_id_type_deadline RENAME TO ix_items_user_id_type_date"
    )
    op.alter_column("items", "deadline", new_column_name="date")

    op.add_column("items", sa.Column("status", sa.String(length=16), nullable=True))
    op.add_column(
        "items",
        sa.Column("solutions", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    )

    op.execute(
        "UPDATE items SET status = CASE WHEN completed THEN 'done' ELSE 'not started' END "
        "WHERE type = 'todo'"
    )

    op.create_check_constraint(
        op.f("ck_items_note_has_no_todo_fields"),
        "items",
        "type <> 'note' OR (date IS NULL AND completed IS NULL AND status IS NULL AND solutions IS NULL AND time_to_complete IS NULL)",
    )
    op.create_check_constraint(
        op.f("ck_items_status_valid"),
        "items",
        "status IN ('not started', 'in progress', 'done', 'archived')",
    )
    op.create_check_constraint(
        op.f("ck_items_todo_status_matches_completed"),
        "items",
        "type <> 'todo' OR (status IS NOT NULL OR completed IS NOT NULL)",
    )
