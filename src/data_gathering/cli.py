import argparse
from data_scraping import download_yarnspirations
from data_upload import upload_folder_to_gcs

def main():
    # Initialize argument parser
    parser = argparse.ArgumentParser(description="A CLI tool for data scraping and uploading to GCS")

    # Add subcommands: 'scrape' for scraping, 'upload' for uploading
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Subcommand for scraping data
    scrape_parser = subparsers.add_parser("scrape", help="Scrape data from Yarnspirations")
    scrape_parser.add_argument("--project-type", type=str, required=True, help="Project type (e.g., Rugs, Scarves, Blankets)")
    scrape_parser.add_argument("--pages", type=int, required=True, help="Number of pages to scrape")

    # Subcommand for uploading data
    upload_parser = subparsers.add_parser("upload", help="Upload data to GCS")
    upload_parser.add_argument("--folder", type=str, required=True, help="Folder to upload (default: /app/input_file)")
    upload_parser.add_argument("--bucket", type=str, required=True, help="GCS bucket name")
    upload_parser.add_argument("--prefix", type=str, default="", help="Optional prefix for blob names in GCS")

    # Parse arguments
    args = parser.parse_args()

    # Handle 'scrape' command
    if args.command == "scrape":
        print(f"Scraping {args.pages} pages of {args.project_type} from Yarnspirations...")
        download_yarnspirations(args.project_type, args.pages)

    # Handle 'upload' command
    elif args.command == "upload":
        folder = args.folder if args.folder else "/app/input_file"  # Use default folder in container if not provided
        print(f"Uploading folder {folder} to GCS bucket {args.bucket} ...")
        upload_folder_to_gcs(args.bucket, folder, args.prefix)

if __name__ == "__main__":
    main()

# Example usage:
# python cli.py scrape --project-type "Rugs" --pages 3
# python3 cli.py upload --folder "/app/input_file" --bucket "crochet-patterns-bucket"