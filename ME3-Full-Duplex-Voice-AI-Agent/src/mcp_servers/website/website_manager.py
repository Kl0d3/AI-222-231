
import html
import ipaddress
import os
import re
import shutil
import socket
import tempfile
import urllib.error
import urllib.parse
import urllib.request

from pathlib import Path, PurePosixPath


# ============================================================
# PATH CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

WEBSITES_ROOT = PROJECT_ROOT / "outputs" / "websites"

MAX_FILE_SIZE = 1_000_000  # 1 MB for website text files
MAX_IMAGE_SIZE = 5_000_000  # 5 MB per downloaded image

DOWNLOAD_TIMEOUT = 20

ALLOWED_EXTENSIONS = {
    ".html",
    ".htm",
    ".css",
    ".js",
    ".mjs",
    ".json",
    ".svg",
    ".xml",
    ".txt",
    ".webmanifest",
}

IMAGE_TYPES = {
    "image/png": {
        "extension": ".png",
        "signature": lambda data: data.startswith(
            b"\x89PNG\r\n\x1a\n"
        ),
    },
    "image/jpeg": {
        "extension": ".jpg",
        "signature": lambda data: data.startswith(
            b"\xff\xd8\xff"
        ),
    },
    "image/webp": {
        "extension": ".webp",
        "signature": lambda data: (
            len(data) >= 12
            and data[:4] == b"RIFF"
            and data[8:12] == b"WEBP"
        ),
    },
    "image/gif": {
        "extension": ".gif",
        "signature": lambda data: data.startswith(
            (b"GIF87a", b"GIF89a")
        ),
    },
}


# ============================================================
# PROJECT NAME VALIDATION
# ============================================================

def normalize_project_name(project_name):

    if not isinstance(project_name, str):
        raise ValueError("Project name must be a string.")

    project_name = project_name.strip()

    if not project_name:
        raise ValueError("Project name cannot be empty.")

    if "/" in project_name or "\\" in project_name:
        raise ValueError(
            "Project name cannot contain path separators."
        )

    if project_name in {".", ".."}:
        raise ValueError("Invalid project name.")

    if project_name.startswith("."):
        raise ValueError(
            "Project name cannot start with a period."
        )

    normalized = re.sub(
        r"[^a-zA-Z0-9]+",
        "_",
        project_name
    ).strip("_").lower()

    if not normalized:
        raise ValueError(
            "Project name must contain letters or numbers."
        )

    if len(normalized) > 64:
        raise ValueError(
            "Project name cannot exceed 64 characters."
        )

    return normalized


# ============================================================
# PROJECT PATH VALIDATION
# ============================================================

def get_project_path(project_name):

    normalized_name = normalize_project_name(project_name)

    root = WEBSITES_ROOT.resolve()

    project_path = (
        root / normalized_name
    ).resolve()

    # Backward compatibility for existing hyphenated projects.
    if not project_path.exists() and "_" in normalized_name:

        legacy_name = normalized_name.replace("_", "-")

        legacy_path = (
            root / legacy_name
        ).resolve()

        if (
            legacy_path.is_relative_to(root)
            and legacy_path.is_dir()
            and not legacy_path.is_symlink()
        ):
            project_path = legacy_path

    if not project_path.is_relative_to(root):
        raise ValueError(
            "Project path is outside the websites directory."
        )

    return normalized_name, project_path


# ============================================================
# WEBSITE FILE PATH VALIDATION
# ============================================================

def get_website_file_path(
    project_name,
    relative_path
):

    normalized_name, project_path = get_project_path(
        project_name
    )

    if not isinstance(relative_path, str):
        raise ValueError("File path must be a string.")

    relative_path = relative_path.strip()

    if not relative_path:
        raise ValueError("File path cannot be empty.")

    relative_path = relative_path.replace("\\", "/")

    file_path = PurePosixPath(relative_path)

    if file_path.is_absolute():
        raise ValueError(
            "Absolute file paths are not allowed."
        )

    if ":" in relative_path:
        raise ValueError(
            "Drive-qualified paths are not allowed."
        )

    if any(
        part in {"", ".", ".."} or part.startswith(".")
        for part in file_path.parts
    ):
        raise ValueError(
            "Invalid or hidden path components are not allowed."
        )

    if not file_path.parts:
        raise ValueError("Invalid file path.")

    extension = Path(file_path.name).suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file extension: {extension}"
        )

    resolved_file_path = (
        project_path.joinpath(*file_path.parts).resolve()
    )

    if not resolved_file_path.is_relative_to(project_path):
        raise ValueError(
            "File path is outside the website project."
        )

    return normalized_name, project_path, resolved_file_path


# ============================================================
# URL AND NETWORK SECURITY
# ============================================================

def validate_download_url(url):

    if not isinstance(url, str):
        raise ValueError("Image URL must be a string.")

    url = url.strip()

    if not url or len(url) > 2048:
        raise ValueError("Invalid image URL.")

    parsed = urllib.parse.urlsplit(url)

    if parsed.scheme.lower() != "https":
        raise ValueError("Only HTTPS image URLs are allowed.")

    if not parsed.hostname:
        raise ValueError("Image URL must contain a hostname.")

    if parsed.username or parsed.password:
        raise ValueError(
            "URLs containing credentials are not allowed."
        )

    if parsed.port not in {None, 443}:
        raise ValueError(
            "Only the standard HTTPS port is allowed."
        )

    hostname = parsed.hostname.rstrip(".").lower()

    if hostname == "localhost" or hostname.endswith(".localhost"):
        raise ValueError("Localhost addresses are not allowed.")

    try:
        ipaddress.ip_address(hostname)
        is_ip_address = True
    except ValueError:
        is_ip_address = False

    if is_ip_address:
        addresses = [hostname]
    else:
        try:
            address_records = socket.getaddrinfo(
                hostname,
                443,
                type=socket.SOCK_STREAM
            )
        except socket.gaierror:
            raise ValueError(
                "Could not resolve the image URL hostname."
            )

        addresses = list({
            record[4][0]
            for record in address_records
        })

    if not addresses:
        raise ValueError(
            "Image URL did not resolve to an IP address."
        )

    for address in addresses:

        parsed_address = ipaddress.ip_address(address)

        if not parsed_address.is_global:
            raise ValueError(
                "Image URLs resolving to private or "
                "non-public addresses are not allowed."
            )

    return url


# ============================================================
# VALIDATED REDIRECT HANDLER
# ============================================================

class SafeRedirectHandler(urllib.request.HTTPRedirectHandler):

    def redirect_request(
        self,
        request,
        response,
        code,
        message,
        headers,
        new_url
    ):

        validate_download_url(new_url)

        return super().redirect_request(
            request,
            response,
            code,
            message,
            headers,
            new_url
        )


# ============================================================
# IMAGE FILENAME VALIDATION
# ============================================================

def normalize_asset_filename(filename, extension):

    if not isinstance(filename, str):
        raise ValueError("Image filename must be a string.")

    filename = filename.strip()

    if not filename:
        filename = "image"

    if "/" in filename or "\\" in filename:
        raise ValueError(
            "Image filename cannot contain path separators."
        )

    if filename in {".", ".."} or filename.startswith("."):
        raise ValueError("Invalid image filename.")

    stem = Path(filename).stem

    if not stem:
        raise ValueError("Image filename must have a name.")

    stem = re.sub(
        r"[^a-zA-Z0-9_-]+",
        "_",
        stem
    ).strip("_")

    if not stem:
        raise ValueError(
            "Image filename must contain letters or numbers."
        )

    if len(stem) > 100:
        stem = stem[:100]

    supplied_extension = Path(filename).suffix.lower()

    if supplied_extension and supplied_extension != extension:
        raise ValueError(
            "Image filename extension does not match "
            "the downloaded image format."
        )

    return f"{stem}{extension}"


# ============================================================
# DOWNLOAD WEBSITE IMAGE ASSET
# ============================================================

def download_website_asset(
    project_name,
    url,
    filename="image"
):

    temporary_path = None

    try:

        normalized_name, project_path = get_project_path(
            project_name
        )

        if not project_path.is_dir():
            return {
                "success": False,
                "error": "Website project does not exist."
            }

        if project_path.is_symlink():
            return {
                "success": False,
                "error": "Symbolic-link projects are not supported."
            }

        assets_path = project_path / "assets"

        assets_path.mkdir(
            parents=True,
            exist_ok=True
        )

        assets_path = assets_path.resolve()

        if not assets_path.is_relative_to(project_path):
            return {
                "success": False,
                "error": "Asset directory is outside the website project."
            }

        if assets_path.is_symlink():
            return {
                "success": False,
                "error": "Symbolic-link asset directories are not allowed."
            }

        safe_url = validate_download_url(url)

        request = urllib.request.Request(
            safe_url,
            headers={
                "User-Agent": "AI-231-Website-Asset-Downloader/1.0",
                "Accept": "image/png,image/jpeg,image/webp,image/gif",
            }
        )

        opener = urllib.request.build_opener(
            SafeRedirectHandler()
        )

        with opener.open(
            request,
            timeout=DOWNLOAD_TIMEOUT
        ) as response:

            # Validate the final URL after redirects.
            validate_download_url(response.geturl())

            content_type = (
                response.headers.get(
                    "Content-Type",
                    ""
                )
                .split(";")[0]
                .strip()
                .lower()
            )

            if content_type not in IMAGE_TYPES:
                raise ValueError(
                    f"Unsupported image content type: {content_type}"
                )

            content_length = response.headers.get(
                "Content-Length"
            )

            if content_length:

                try:
                    declared_size = int(content_length)
                except ValueError:
                    raise ValueError(
                        "Invalid image Content-Length header."
                    )

                if declared_size < 0 or declared_size > MAX_IMAGE_SIZE:
                    raise ValueError(
                        "Image exceeds the 5 MB download limit."
                    )

            image_data = response.read(
                MAX_IMAGE_SIZE + 1
            )

            if len(image_data) > MAX_IMAGE_SIZE:
                raise ValueError(
                    "Image exceeds the 5 MB download limit."
                )

        if not image_data:
            raise ValueError("Downloaded image is empty.")

        image_info = IMAGE_TYPES[content_type]

        if not image_info["signature"](image_data):
            raise ValueError(
                "Downloaded file does not match its declared image format."
            )

        extension = image_info["extension"]

        safe_filename = normalize_asset_filename(
            filename,
            extension
        )

        destination = assets_path / safe_filename

        resolved_destination = destination.resolve()

        if not resolved_destination.is_relative_to(assets_path):
            raise ValueError(
                "Image destination is outside the assets directory."
            )

        if destination.exists() or destination.is_symlink():
            return {
                "success": False,
                "error": (
                    f"Asset '{safe_filename}' already exists. "
                    "Choose another filename to avoid overwriting it."
                )
            }

        # Write to a temporary file first, then atomically rename.
        file_descriptor, temporary_name = tempfile.mkstemp(
            prefix=".asset_",
            suffix=".tmp",
            dir=assets_path
        )

        temporary_path = Path(temporary_name)

        with os.fdopen(file_descriptor, "wb") as output_file:
            output_file.write(image_data)

        # Recheck the destination before publishing the file.
        if destination.exists() or destination.is_symlink():
            return {
                "success": False,
                "error": (
                    f"Asset '{safe_filename}' already exists."
                )
            }

        temporary_path.replace(destination)
        temporary_path = None

        return {
            "success": True,
            "project_name": normalized_name,
            "relative_path": f"assets/{safe_filename}",
            "filename": safe_filename,
            "content_type": content_type,
            "size_bytes": len(image_data),
            "message": "Website image asset downloaded successfully."
        }

    except urllib.error.HTTPError as error:

        return {
            "success": False,
            "error": (
                f"Image server returned HTTP status {error.code}."
            )
        }

    except urllib.error.URLError:
        return {
            "success": False,
            "error": "Could not connect to the image server."
        }

    except Exception as error:

        return {
            "success": False,
            "error": str(error)
        }

    finally:

        if temporary_path is not None:

            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass


# ============================================================
# CREATE WEBSITE PROJECT
# ============================================================

def create_website_project(
    project_name,
    description=""
):

    try:

        normalized_name, project_path = get_project_path(
            project_name
        )

        WEBSITES_ROOT.mkdir(
            parents=True,
            exist_ok=True
        )

        if project_path.exists():
            return {
                "success": False,
                "error": (
                    f"Website project '{normalized_name}' "
                    "already exists."
                )
            }

        project_path.mkdir(
            parents=False,
            exist_ok=False
        )

        assets_path = project_path / "assets"
        assets_path.mkdir()

        safe_title = html.escape(
            normalized_name.replace("_", " ").title()
        )

        index_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta name="description" content="{html.escape(description)}">
    <title>{safe_title}</title>
    <link rel="stylesheet" href="style.css">
</head>
<body>

    <main>
        <h1>{safe_title}</h1>
        <p>Website content will be generated here.</p>
    </main>

    <script src="script.js"></script>
</body>
</html>
"""

        style_content = """* {
    box-sizing: border-box;
}

body {
    margin: 0;
    font-family: Arial, sans-serif;
    line-height: 1.6;
}
"""

        script_content = """// Website JavaScript
"""

        (project_path / "index.html").write_text(
            index_content,
            encoding="utf-8"
        )

        (project_path / "style.css").write_text(
            style_content,
            encoding="utf-8"
        )

        (project_path / "script.js").write_text(
            script_content,
            encoding="utf-8"
        )

        return {
            "success": True,
            "project_name": normalized_name,
            "project_path": str(project_path),
            "files_created": [
                "index.html",
                "style.css",
                "script.js",
                "assets/"
            ],
            "message": "Website project created successfully."
        }

    except Exception as error:

        return {
            "success": False,
            "error": str(error)
        }


# ============================================================
# LIST WEBSITE PROJECTS
# ============================================================

def list_website_projects():

    try:

        if not WEBSITES_ROOT.exists():
            return {
                "success": True,
                "projects": [],
                "count": 0
            }

        projects = []

        for project_path in sorted(
            WEBSITES_ROOT.iterdir()
        ):

            if not project_path.is_dir():
                continue

            if project_path.is_symlink():
                continue

            if project_path.name.startswith("."):
                continue

            projects.append({
                "project_name": project_path.name,
                "has_index": (
                    project_path / "index.html"
                ).is_file(),
                "has_assets": (
                    project_path / "assets"
                ).is_dir()
            })

        return {
            "success": True,
            "projects": projects,
            "count": len(projects)
        }

    except Exception as error:

        return {
            "success": False,
            "error": str(error)
        }


# ============================================================
# READ WEBSITE FILE
# ============================================================

def read_website_file(
    project_name,
    relative_path
):

    try:

        normalized_name, project_path, file_path = (
            get_website_file_path(
                project_name,
                relative_path
            )
        )

        if not project_path.is_dir():
            return {
                "success": False,
                "error": "Website project does not exist."
            }

        if not file_path.is_file():
            return {
                "success": False,
                "error": "Website file does not exist."
            }

        if file_path.stat().st_size > MAX_FILE_SIZE:
            return {
                "success": False,
                "error": "File exceeds the 1 MB read limit."
            }

        content = file_path.read_text(
            encoding="utf-8"
        )

        return {
            "success": True,
            "project_name": normalized_name,
            "relative_path": relative_path,
            "content": content
        }

    except Exception as error:

        return {
            "success": False,
            "error": str(error)
        }


# ============================================================
# CREATE OR UPDATE WEBSITE FILE
# ============================================================

def write_website_file(
    project_name,
    relative_path,
    content
):

    try:

        normalized_name, project_path, file_path = (
            get_website_file_path(
                project_name,
                relative_path
            )
        )

        if not project_path.is_dir():
            return {
                "success": False,
                "error": "Website project does not exist."
            }

        if not isinstance(content, str):
            return {
                "success": False,
                "error": "File content must be a string."
            }

        content_size = len(content.encode("utf-8"))

        if content_size > MAX_FILE_SIZE:
            return {
                "success": False,
                "error": "File exceeds the 1 MB write limit."
            }

        file_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        resolved_path = file_path.resolve()

        if not resolved_path.is_relative_to(project_path):
            return {
                "success": False,
                "error": (
                    "File path is outside the website project."
                )
            }

        existed = resolved_path.exists()

        resolved_path.write_text(
            content,
            encoding="utf-8"
        )

        return {
            "success": True,
            "project_name": normalized_name,
            "relative_path": relative_path,
            "operation": "updated" if existed else "created",
            "size_bytes": content_size,
            "message": "Website file saved successfully."
        }

    except Exception as error:

        return {
            "success": False,
            "error": str(error)
        }


# ============================================================
# DELETE WEBSITE PROJECT
# ============================================================

def delete_website_project(project_name):

    try:

        normalized_name, project_path = get_project_path(
            project_name
        )

        if not project_path.exists():
            return {
                "success": False,
                "error": "Website project does not exist."
            }

        if project_path.is_symlink():
            return {
                "success": False,
                "error": (
                    "Symbolic-link projects cannot be deleted."
                )
            }

        if not project_path.is_dir():
            return {
                "success": False,
                "error": (
                    "Target is not a website project directory."
                )
            }

        shutil.rmtree(project_path)

        return {
            "success": True,
            "project_name": normalized_name,
            "message": "Website project deleted successfully."
        }

    except Exception as error:

        return {
            "success": False,
            "error": str(error)
        }
