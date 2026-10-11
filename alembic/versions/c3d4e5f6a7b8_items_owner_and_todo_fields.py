"""items: owner column and agent todo fields

Items become the chat agent's long-term memory, so they need an owner, and todos gain
the two fields the agent extracts (solutions, time_to_complete).

Existing rows are assigned to the user 'default', the out-of-the-box value of
CUEAI_DEFAULT_USER_ID (what requests without an X-User-Id header act as).

Revision ID: c3d4e5f6a7b8
Revises: b7c1d2e3f4a5
Create Date: 2026-09-26 12:00:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = "c3d4e5f6a7b8"
down_revision: Union[str, Sequence[str], None] = "b7c1d2e3f4a5"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_OLD_NOTE_CHECK = "type <> 'note' OR (date IS NULL AND completed IS NULL)"
_NEW_NOTE_CHECK = (
    "type <> 'note' OR (date IS NULL AND completed IS NULL "
    "AND solutions IS NULL AND time_to_complete IS NULL)"
)


def upgrade() -> None:
    """Upgrade schema."""
    # Backfill existing rows through a temporary server default, then drop it: the
    # application always sets the owner explicitly.
    op.add_column(
        "items",
        sa.Column("user_id", sa.String(length=128), nullable=False, server_default="default"),
    )
    op.alter_column("items", "user_id", server_default=None)

    op.add_column(
        "items", sa.Column("solutions", postgresql.JSONB(astext_type=sa.Text()), nullable=True)
    )
    op.add_column("items", sa.Column("time_to_complete", sa.Integer(), nullable=True))

    op.drop_constraint(op.f("ck_items_note_has_no_todo_fields"), "items", type_="check")
    op.create_check_constraint(op.f("ck_items_note_has_no_todo_fields"), "items", _NEW_NOTE_CHECK)
    op.create_check_constraint(
        op.f("ck_items_time_to_complete_positive"),
        "items",
        "time_to_complete IS NULL OR time_to_complete > 0",
    )

    op.drop_index("ix_items_type_date", table_name="items")
    op.create_index("ix_items_user_id_type_date", "items", ["user_id", "type", "date"])


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index("ix_items_user_id_type_date", table_name="items")
    op.create_index("ix_items_type_date", "items", ["type", "date"])

    op.drop_constraint(op.f("ck_items_time_to_complete_positive"), "items", type_="check")
    op.drop_constraint(op.f("ck_items_note_has_no_todo_fields"), "items", type_="check")
    op.create_check_constraint(op.f("ck_items_note_has_no_todo_fields"), "items", _OLD_NOTE_CHECK)

    op.drop_column("items", "time_to_complete")
    op.drop_column("items", "solutions")
    op.drop_column("items", "user_id")
