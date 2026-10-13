"""Publish a new classifier so the app can download it (model auto-update).

Usage (from the repo root, with the same AWS env vars as the API):
    python ml/publish_model.py ml/plantsnap_v1.mlpackage 1.1

Uploads every file in the .mlpackage to
    s3://$S3_BUCKET_NAME/models/<version>/plantsnap.mlpackage/<file>
Then set MODEL_VERSION=<version> on Render. Phones pick it up on next launch.
"""
import os
import sys
from pathlib import Path

import boto3


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    package, version = Path(sys.argv[1]), sys.argv[2]
    if not (package.is_dir() and package.suffix == ".mlpackage"):
        sys.exit(f"Not an .mlpackage folder: {package}")

    bucket = os.environ["S3_BUCKET_NAME"]
    s3 = boto3.client("s3", region_name=os.getenv("AWS_REGION", "us-west-2"))
    prefix = f"models/{version}/plantsnap.mlpackage/"

    files = [f for f in package.rglob("*") if f.is_file()]
    for f in files:
        key = prefix + f.relative_to(package).as_posix()
        print(f"⬆️  {key} ({f.stat().st_size / 1e6:.1f} MB)")
        s3.upload_file(str(f), bucket, key)

    print(f"\n✅ Uploaded {len(files)} files for version {version}.")
    print(f"Next: set MODEL_VERSION={version} on Render (Environment tab). Phones update on next launch.")


if __name__ == "__main__":
    main()
