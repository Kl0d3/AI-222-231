
from pathlib import Path

from src.mcp_servers.website.website_manager import (
    create_website_project,
    list_website_projects,
    read_website_file,
    write_website_file,
    download_website_asset,
    delete_website_project,
)


def run_tests():

    project_name = "website-manager-test"
    asset_test_project = "website_asset_download_test"

    print("\nWEBSITE MANAGER TESTS")
    print("=" * 55)

    # ========================================================
    # EXISTING WEBSITE MANAGER TESTS
    # ========================================================

    # Test 1: Create project
    result = create_website_project(
        project_name,
        "Website manager test project."
    )

    # The existing project may already exist.
    if not result["success"]:
        assert "already exists" in result["error"].lower(), result

    print("PASS: Website project creation/existence")

    # Test 2: List projects
    result = list_website_projects()

    assert result["success"], result

    assert any(
        project["project_name"] == project_name
        for project in result["projects"]
    )

    print("PASS: Website project listing")

    # Test 3: Read index.html
    result = read_website_file(
        project_name,
        "index.html"
    )

    assert result["success"], result

    assert "<!DOCTYPE html>" in result["content"]

    print("PASS: Website file reading")

    # Test 4: Update index.html
    new_content = """<!DOCTYPE html>
<html>
<head>
    <title>Updated Test Website</title>
</head>
<body>
    <h1>Website manager test passed.</h1>
</body>
</html>
"""

    result = write_website_file(
        project_name,
        "index.html",
        new_content
    )

    assert result["success"], result

    assert result["operation"] == "updated"

    print("PASS: Website file writing")

    # Test 5: Verify updated content
    result = read_website_file(
        project_name,
        "index.html"
    )

    assert result["success"], result

    assert "Website manager test passed." in result["content"]

    print("PASS: Updated content verification")

    # Test 6: Reject path traversal
    result = read_website_file(
        project_name,
        "../../.env"
    )

    assert not result["success"]

    print("PASS: Path traversal protection")

    # ========================================================
    # IMAGE DOWNLOAD TESTS
    # ========================================================

    print("\nIMAGE DOWNLOAD TESTS")
    print("-" * 55)

    try:

        # Test 7: Create isolated image test project
        result = create_website_project(
            asset_test_project,
            "Temporary image download test."
        )

        assert result["success"], result

        print("PASS: Image test project creation")

        # Test 8: Download a real PNG image
        image_url = (
            "https://www.python.org/static/community_logos/"
            "python-logo.png"
        )

        result = download_website_asset(
            asset_test_project,
            image_url,
            "python_logo"
        )

        assert result["success"], result

        assert result["relative_path"] == (
            "assets/python_logo.png"
        )

        assert result["content_type"] == "image/png"

        print("PASS: PNG image download")

        # Test 9: Verify downloaded file
        image_path = (
            Path(result["relative_path"])
        )

        # Resolve the actual project directory.
        from src.mcp_servers.website.website_manager import (
            get_project_path,
        )

        _, project_path = get_project_path(
            asset_test_project
        )

        image_path = project_path / result["relative_path"]

        assert image_path.is_file()

        image_bytes = image_path.read_bytes()

        assert image_bytes.startswith(
            b"\x89PNG\r\n\x1a\n"
        )

        assert len(image_bytes) <= 5_000_000

        print("PASS: Downloaded image integrity")

        # Test 10: Prevent accidental overwrite
        result = download_website_asset(
            asset_test_project,
            image_url,
            "python_logo"
        )

        assert not result["success"]

        assert "already exists" in result["error"].lower()

        print("PASS: Existing asset overwrite protection")

        # Test 11: Reject unsafe URL
        result = download_website_asset(
            asset_test_project,
            "http://example.com/image.png",
            "unsafe_image"
        )

        assert not result["success"]

        print("PASS: Unsafe image URL rejection")

    finally:

        # Remove only the temporary test project.
        cleanup_result = delete_website_project(
            asset_test_project
        )

        if cleanup_result["success"]:
            print("PASS: Temporary image test cleanup")

        elif "does not exist" not in cleanup_result.get(
            "error", ""
        ).lower():
            print(
                "WARNING: Temporary test cleanup failed:",
                cleanup_result
            )

    print("\n" + "=" * 55)
    print("ALL WEBSITE MANAGER TESTS PASSED")


if __name__ == "__main__":
    run_tests()
