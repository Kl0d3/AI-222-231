
import os
import json
import base64
import urllib.request
import urllib.error
import urllib.parse

from pathlib import Path
from datetime import datetime, timezone

from dotenv import load_dotenv

from src.mcp_servers.website.website_manager import (
    WEBSITES_ROOT,
    get_project_path,
    ALLOWED_EXTENSIONS,
)


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

load_dotenv(PROJECT_ROOT / ".env")

GITHUB_API = "https://api.github.com"

GITHUB_API_VERSION = "2022-11-28"

REQUEST_TIMEOUT = 30

METADATA_PATH = WEBSITES_ROOT / ".agent_metadata.json"

MAX_TEXT_FILE_SIZE = 1_000_000

MAX_IMAGE_FILE_SIZE = 5_000_000

IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".gif",
}

IMAGE_SIGNATURES = {
    ".png": lambda data: data.startswith(
        b"\x89PNG\r\n\x1a\n"
    ),
    ".jpg": lambda data: data.startswith(
        b"\xff\xd8\xff"
    ),
    ".jpeg": lambda data: data.startswith(
        b"\xff\xd8\xff"
    ),
    ".webp": lambda data: (
        len(data) >= 12
        and data.startswith(b"RIFF")
        and data[8:12] == b"WEBP"
    ),
    ".gif": lambda data: (
        data.startswith(b"GIF87a")
        or data.startswith(b"GIF89a")
    ),
}


# ============================================================
# GITHUB API ERROR
# ============================================================

class GitHubAPIError(Exception):

    def __init__(self, status_code, message):

        self.status_code = status_code
        self.message = message

        super().__init__(
            f"GitHub API error ({status_code}): {message}"
        )


# ============================================================
# AUTHENTICATION
# ============================================================

def get_github_token():

    token = os.getenv("GITHUB_TOKEN")

    if not token:

        raise RuntimeError(
            "GITHUB_TOKEN was not found in the project .env file."
        )

    return token


def get_headers():

    return {
        "Authorization": f"Bearer {get_github_token()}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": GITHUB_API_VERSION,
        "Content-Type": "application/json",
    }


# ============================================================
# HTTP REQUEST HANDLER
# ============================================================

def github_request(
    method,
    endpoint,
    payload=None
):

    url = f"{GITHUB_API}{endpoint}"

    data = None

    if payload is not None:

        data = json.dumps(payload).encode("utf-8")

    request = urllib.request.Request(
        url,
        data=data,
        headers=get_headers(),
        method=method,
    )

    try:

        with urllib.request.urlopen(
            request,
            timeout=REQUEST_TIMEOUT
        ) as response:

            response_data = response.read()

            if not response_data:

                return {
                    "status_code": response.status
                }

            return json.loads(
                response_data.decode("utf-8")
            )

    except urllib.error.HTTPError as error:

        try:

            error_data = json.loads(
                error.read().decode("utf-8")
            )

            message = error_data.get(
                "message",
                "Unknown GitHub API error."
            )

        except Exception:

            message = (
                "GitHub returned an unreadable error response."
            )

        raise GitHubAPIError(
            error.code,
            message
        )

    except urllib.error.URLError as error:

        raise RuntimeError(
            f"Could not connect to GitHub: {error.reason}"
        )


# ============================================================
# LOCAL METADATA MANAGEMENT
# ============================================================

def load_metadata():

    if not METADATA_PATH.exists():

        return {
            "projects": {}
        }

    try:

        with METADATA_PATH.open(
            "r",
            encoding="utf-8"
        ) as file:

            metadata = json.load(file)

        if not isinstance(metadata, dict):

            raise ValueError(
                "Invalid metadata structure."
            )

        if not isinstance(
            metadata.get("projects"),
            dict
        ):

            raise ValueError(
                "Invalid projects metadata."
            )

        return metadata

    except Exception as error:

        raise RuntimeError(
            f"Could not read website metadata: {error}"
        )


def save_metadata(metadata):

    WEBSITES_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    temporary_path = (
        WEBSITES_ROOT / ".agent_metadata.tmp"
    )

    try:

        with temporary_path.open(
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                metadata,
                file,
                indent=4
            )

        temporary_path.replace(
            METADATA_PATH
        )

    finally:

        if temporary_path.exists():

            temporary_path.unlink()


def get_managed_repository(project_name):

    normalized_name, project_path = (
        get_project_path(project_name)
    )

    if not project_path.is_dir():

        raise ValueError(
            "Website project does not exist."
        )

    metadata = load_metadata()

    repository = metadata["projects"].get(
        normalized_name
    )

    if not repository:

        raise ValueError(
            "This website has no GitHub repository "
            "registered as managed by this agent."
        )

    if repository.get("created_by_agent") is not True:

        raise ValueError(
            "This repository is not registered as "
            "managed by this agent."
        )

    return normalized_name, repository


# ============================================================
# GITHUB ACCOUNT
# ============================================================

def get_github_account():

    user = github_request(
        "GET",
        "/user"
    )

    return user["login"]


# ============================================================
# REPOSITORY NAME VALIDATION
# ============================================================

def normalize_repository_name(repository_name):

    if not isinstance(repository_name, str):

        raise ValueError(
            "Repository name must be a string."
        )

    repository_name = repository_name.strip()

    if not repository_name:

        raise ValueError(
            "Repository name cannot be empty."
        )

    if len(repository_name) > 100:

        raise ValueError(
            "Repository name is too long."
        )

    if not all(
        character.isalnum()
        or character in "-_."
        for character in repository_name
    ):

        raise ValueError(
            "Repository name contains unsupported characters."
        )

    if repository_name in {".", ".."}:

        raise ValueError(
            "Invalid repository name."
        )

    return repository_name


# ============================================================
# IMAGE VALIDATION
# ============================================================

def validate_image_content(
    extension,
    content
):

    validator = IMAGE_SIGNATURES.get(
        extension.lower()
    )

    if validator is None:

        raise ValueError(
            f"Unsupported image extension: {extension}"
        )

    if not validator(content):

        raise ValueError(
            f"Image content does not match its extension: {extension}"
        )


# ============================================================
# WEBSITE FILE DISCOVERY
# ============================================================

def collect_website_files(project_name):

    normalized_name, project_path = (
        get_project_path(project_name)
    )

    if not project_path.is_dir():

        raise ValueError(
            "Website project does not exist."
        )

    if project_path.is_symlink():

        raise ValueError(
            "Website project directory cannot be a symbolic link."
        )

    files = []

    def raise_walk_error(error):
        raise error

    # os.walk allows us to explicitly prune hidden and
    # symbolic-link directories before traversing them.
    for root, directories, filenames in os.walk(
        project_path,
        topdown=True,
        followlinks=False,
        onerror=raise_walk_error
    ):

        root_path = Path(root)

        safe_directories = []

        for directory in directories:

            directory_path = root_path / directory

            if directory.startswith("."):

                continue

            if directory_path.is_symlink():

                continue

            safe_directories.append(directory)

        directories[:] = safe_directories

        for filename in filenames:

            file_path = root_path / filename

            if filename.startswith("."):

                continue

            if file_path.is_symlink():

                continue

            if not file_path.is_file():

                continue

            relative_path = file_path.relative_to(
                project_path
            )

            extension = file_path.suffix.lower()

            is_image = extension in IMAGE_EXTENSIONS

            is_text = extension in ALLOWED_EXTENSIONS

            if not is_image and not is_text:

                continue

            maximum_size = (
                MAX_IMAGE_FILE_SIZE
                if is_image
                else MAX_TEXT_FILE_SIZE
            )

            file_size = file_path.stat().st_size

            if file_size > maximum_size:

                raise ValueError(
                    f"File exceeds its size limit: "
                    f"{relative_path} "
                    f"({file_size} bytes, maximum "
                    f"{maximum_size} bytes)"
                )

            # Read all supported files as raw bytes.
            # This preserves both text and binary content.
            content = file_path.read_bytes()

            # Check again after reading in case the file
            # changed between stat() and read_bytes().
            if len(content) > maximum_size:

                raise ValueError(
                    f"File exceeds its size limit: {relative_path}"
                )

            if is_image:

                validate_image_content(
                    extension,
                    content
                )

            files.append({
                "path": relative_path.as_posix(),
                "content": content,
                "file_type": (
                    "image" if is_image else "text"
                ),
                "size": len(content),
            })

    if not files:

        raise ValueError(
            "No supported website files were found."
        )

    if not any(
        item["path"] == "index.html"
        for item in files
    ):

        raise ValueError(
            "Website project must contain index.html."
        )

    return files


# ============================================================
# GET CURRENT BRANCH REFERENCE
# ============================================================

def get_branch_reference(
    owner,
    repository,
    branch
):

    encoded_branch = urllib.parse.quote(
        branch,
        safe=""
    )

    return github_request(
        "GET",
        f"/repos/{owner}/{repository}/git/ref/heads/{encoded_branch}"
    )


# ============================================================
# CREATE GIT BLOB
# ============================================================

def create_git_blob(
    owner,
    repository,
    content
):

    # Accept both bytes and strings.
    # Strings are converted to UTF-8 bytes first.
    if isinstance(content, str):

        content = content.encode("utf-8")

    if not isinstance(content, bytes):

        raise TypeError(
            "Git blob content must be bytes or a string."
        )

    encoded_content = base64.b64encode(
        content
    ).decode("ascii")

    result = github_request(
        "POST",
        f"/repos/{owner}/{repository}/git/blobs",
        {
            "content": encoded_content,
            "encoding": "base64"
        }
    )

    return result["sha"]


# ============================================================
# CREATE GIT TREE
# ============================================================

def create_git_tree(
    owner,
    repository,
    files,
    base_tree_sha=None
):

    tree_entries = []

    for website_file in files:

        blob_sha = create_git_blob(
            owner,
            repository,
            website_file["content"]
        )

        tree_entries.append({
            "path": website_file["path"],
            "mode": "100644",
            "type": "blob",
            "sha": blob_sha
        })

    payload = {
        "tree": tree_entries
    }

    if base_tree_sha:

        payload["base_tree"] = base_tree_sha

    result = github_request(
        "POST",
        f"/repos/{owner}/{repository}/git/trees",
        payload
    )

    if result.get("truncated"):

        raise RuntimeError(
            "GitHub returned a truncated Git tree."
        )

    return result["sha"]


# ============================================================
# CREATE GIT COMMIT
# ============================================================

def create_git_commit(
    owner,
    repository,
    message,
    tree_sha,
    parent_sha
):

    result = github_request(
        "POST",
        f"/repos/{owner}/{repository}/git/commits",
        {
            "message": message,
            "tree": tree_sha,
            "parents": [parent_sha]
        }
    )

    return result["sha"]


# ============================================================
# UPDATE BRANCH REFERENCE
# ============================================================

def update_branch_reference(
    owner,
    repository,
    branch,
    commit_sha
):

    encoded_branch = urllib.parse.quote(
        branch,
        safe=""
    )

    return github_request(
        "PATCH",
        f"/repos/{owner}/{repository}/git/refs/heads/{encoded_branch}",
        {
            "sha": commit_sha,
            "force": False
        }
    )


# ============================================================
# PUBLISH WEBSITE FILES IN ONE COMMIT
# ============================================================

def publish_files_as_single_commit(
    owner,
    repository,
    branch,
    project_name
):

    files = collect_website_files(
        project_name
    )

    # Get the current branch commit.
    reference = get_branch_reference(
        owner,
        repository,
        branch
    )

    parent_sha = reference["object"]["sha"]

    # Retrieve the current commit's tree.
    parent_commit = github_request(
        "GET",
        f"/repos/{owner}/{repository}/git/commits/{parent_sha}"
    )

    base_tree_sha = parent_commit["tree"]["sha"]

    # Create blobs and one new tree.
    # Both text and binary files are supported.
    tree_sha = create_git_tree(
        owner,
        repository,
        files,
        base_tree_sha
    )

    # Create one commit containing all website files.
    commit_sha = create_git_commit(
        owner,
        repository,
        "Publish website update",
        tree_sha,
        parent_sha
    )

    # Move the branch to the new commit.
    update_branch_reference(
        owner,
        repository,
        branch,
        commit_sha
    )

    return {
        "commit_sha": commit_sha,
        "parent_sha": parent_sha,
        "files": [
            item["path"]
            for item in files
        ],
        "file_count": len(files),
        "text_file_count": sum(
            item["file_type"] == "text"
            for item in files
        ),
        "image_file_count": sum(
            item["file_type"] == "image"
            for item in files
        ),
    }


# ============================================================
# ENABLE GITHUB PAGES
# ============================================================

def enable_github_pages(
    owner,
    repository,
    branch
):

    try:

        result = github_request(
            "POST",
            f"/repos/{owner}/{repository}/pages",
            {
                "source": {
                    "branch": branch,
                    "path": "/"
                }
            }
        )

        return {
            "enabled": True,
            "configuration": result
        }

    except GitHubAPIError as error:

        if error.status_code not in {409, 422}:

            raise

        # Pages may already be enabled.
        existing = github_request(
            "GET",
            f"/repos/{owner}/{repository}/pages"
        )

        return {
            "enabled": True,
            "configuration": existing
        }


# ============================================================
# PUBLISH WEBSITE
# ============================================================

def publish_website(
    project_name,
    repository_name=None,
    description="Website created by AI-231 Voice Assistant",
    private=False
):

    repository_created = False

    try:

        normalized_name, project_path = (
            get_project_path(project_name)
        )

        if not project_path.is_dir():

            return {
                "success": False,
                "error": "Website project does not exist."
            }

        if private:

            return {
                "success": False,
                "error": (
                    "GitHub Pages publication is configured "
                    "for public repositories in this version."
                )
            }

        metadata = load_metadata()

        if normalized_name in metadata["projects"]:

            return {
                "success": False,
                "error": (
                    "This website already has a managed "
                    "GitHub repository. Use update_published_website."
                )
            }

        if repository_name is None:

            repository_name = normalized_name

        repository_name = normalize_repository_name(
            repository_name
        )

        owner = get_github_account()

        # Verify that the local website can be published.
        collect_website_files(
            normalized_name
        )

        # ----------------------------------------------------
        # CREATE REPOSITORY
        # ----------------------------------------------------

        repository_data = github_request(
            "POST",
            "/user/repos",
            {
                "name": repository_name,
                "description": description,
                "private": False,
                "auto_init": True
            }
        )

        repository_created = True

        branch = repository_data.get(
            "default_branch",
            "main"
        )

        repository_url = repository_data["html_url"]

        pages_url = (
            f"https://{owner}.github.io/{repository_name}/"
        )

        # Register immediately so a failed publication
        # can be recovered through update_published_website.
        metadata["projects"][normalized_name] = {
            "owner": owner,
            "repository": repository_name,
            "branch": branch,
            "repository_url": repository_url,
            "pages_url": pages_url,
            "created_by_agent": True,
            "created_at": datetime.now(
                timezone.utc
            ).isoformat()
        }

        save_metadata(metadata)

        # ----------------------------------------------------
        # PUBLISH ALL FILES IN ONE COMMIT
        # ----------------------------------------------------

        commit_result = publish_files_as_single_commit(
            owner,
            repository_name,
            branch,
            normalized_name
        )

        # ----------------------------------------------------
        # ENABLE GITHUB PAGES
        # ----------------------------------------------------

        pages_result = enable_github_pages(
            owner,
            repository_name,
            branch
        )

        return {
            "success": True,
            "repository_created": True,
            "repository": repository_name,
            "owner": owner,
            "repository_url": repository_url,
            "website_url": pages_url,
            "branch": branch,
            "commit": commit_result,
            "pages": pages_result,
            "message": (
                "Website published in one Git commit. "
                "GitHub Pages deployment may take a few minutes."
            )
        }

    except Exception as error:

        return {
            "success": False,
            "repository_created": repository_created,
            "error": str(error),
            "message": (
                "Publication did not complete successfully. "
                "If the repository was created, its local metadata "
                "is retained so publication can be retried."
            )
        }


# ============================================================
# UPDATE PUBLISHED WEBSITE
# ============================================================

def update_published_website(project_name):

    try:

        normalized_name, repository_data = (
            get_managed_repository(project_name)
        )

        owner = repository_data["owner"]

        repository = repository_data["repository"]

        branch = repository_data["branch"]

        commit_result = publish_files_as_single_commit(
            owner,
            repository,
            branch,
            normalized_name
        )

        pages_result = enable_github_pages(
            owner,
            repository,
            branch
        )

        return {
            "success": True,
            "repository": repository,
            "repository_url": repository_data["repository_url"],
            "website_url": repository_data["pages_url"],
            "commit": commit_result,
            "pages": pages_result,
            "message": (
                "Website updated in one Git commit. "
                "GitHub Pages may take a few minutes to rebuild."
            )
        }

    except Exception as error:

        return {
            "success": False,
            "error": str(error)
        }


# ============================================================
# WEBSITE PUBLICATION STATUS
# ============================================================

def get_website_status(project_name):

    try:

        normalized_name, repository_data = (
            get_managed_repository(project_name)
        )

        owner = repository_data["owner"]

        repository = repository_data["repository"]

        pages_status = None

        latest_build = None

        # ----------------------------------------------------
        # GET PAGES CONFIGURATION
        # ----------------------------------------------------

        try:

            pages_status = github_request(
                "GET",
                f"/repos/{owner}/{repository}/pages"
            )

        except GitHubAPIError as error:

            if error.status_code != 404:

                raise

        # ----------------------------------------------------
        # GET LATEST PAGES BUILD
        # ----------------------------------------------------

        try:

            latest_build = github_request(
                "GET",
                f"/repos/{owner}/{repository}/pages/builds/latest"
            )

        except GitHubAPIError as error:

            if error.status_code != 404:

                raise

        return {
            "success": True,
            "project_name": normalized_name,
            "repository": repository,
            "repository_url": repository_data["repository_url"],
            "website_url": repository_data["pages_url"],
            "pages_enabled": pages_status is not None,
            "pages_status": pages_status,
            "latest_build": latest_build
        }

    except Exception as error:

        return {
            "success": False,
            "error": str(error)
        }
