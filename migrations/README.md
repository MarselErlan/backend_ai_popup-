# Database Migrations

This directory contains database migration scripts for the AI Popup application.

## Resume Documents Table Migration

The `migrate_resume_table.py` script simplifies the resume_documents table by:

1. Removing unnecessary columns:

   - Removed: file_content, content_type, file_size, is_active
   - Removed: created_at, updated_at, last_processed_at

2. Keeping only essential columns:
   - id (SERIAL PRIMARY KEY)
   - user_id (VARCHAR(100))
   - filename (VARCHAR(255))
   - processing_status (VARCHAR(50))

### Migration Process

The script performs the following steps:

1. Creates a backup of existing data
2. Drops the existing table
3. Creates new table with simplified schema
4. Creates index on user_id
5. Restores data from backup
6. Resets the ID sequence
7. Cleans up backup table

### Running the Migration

1. Make sure the application is not running
2. Backup your database (recommended)
3. Run the migration:

```bash
cd /path/to/backend_ai_popup
python3 migrations/migrate_resume_table.py
```

### Rollback

If you need to rollback:

1. The backup table is automatically cleaned up after successful migration
2. If migration fails, the backup table (`resume_documents_backup`) will be preserved
3. You can manually restore from your database backup
