"""items: todo status

Todos gain the four-value status the chat agent tracks ("not started", "in progress",
"done", "archived"). ``completed`` stays, kept equal to ``status = 'done'`` by a CHECK
constraint, so clients that only know the boolean keep working.

Existing todos are backfilled from ``completed``: true becomes "done", anything else
"not started".

Revision ID: d4e5f6a7b8c9
Revises: c3d4e5f6a7b8
Create Date: 2026-09-26 18:00:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "d4e5f6a7b8c9"
down_revision: Union[str, Sequence[str], None] = "c3d4e5f6a7b8"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_OLD_NOTE_CHECK = (
    "type <> 'note' OR (date IS NULL AND completed IS NULL "
    "AND solutions IS NULL AND time_to_complete IS NULL)"
)
_NEW_NOTE_CHECK = (
    "type <> 'note' OR (date IS NULL AND completed IS NULL AND status IS NULL "
    "AND solutions IS NULL AND time_to_complete IS NULL)"
)


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column("items", sa.Column("status", sa.String(length=16), nullable=True))

    op.execute(
        """
        UPDATE items
           SET completed = COALESCE(completed, false),
               status = CASE WHEN completed THEN 'done' ELSE 'not started' END
         WHERE type = 'todo'
        """
    )

    op.drop_constraint(op.f("ck_items_note_has_no_todo_fields"), "items", type_="check")
    op.create_check_constraint(op.f("ck_items_note_has_no_todo_fields"), "items", _NEW_NOTE_CHECK)
    op.create_check_constraint(
        op.f("ck_items_status_valid"),
        "items",
        "status IS NULL OR status IN ('not started', 'in progress', 'done', 'archived')",
    )
    op.create_check_constraint(
        op.f("ck_items_todo_status_matches_completed"),
        "items",
        "type <> 'todo' OR (status IS NOT NULL AND completed IS NOT NULL "
        "AND completed = (status = 'done'))",
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_constraint(op.f("ck_items_todo_status_matches_completed"), "items", type_="check")
    op.drop_constraint(op.f("ck_items_status_valid"), "items", type_="check")
    op.drop_constraint(op.f("ck_items_note_has_no_todo_fields"), "items", type_="check")
    op.create_check_constraint(op.f("ck_items_note_has_no_todo_fields"), "items", _OLD_NOTE_CHECK)
    op.drop_column("items", "status")
