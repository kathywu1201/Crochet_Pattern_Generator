import pytest
from pytest_docker_tools import DockerContainer, docker_image
import os
import time

# Define constants for image and container names
IMAGE_TAG = "llm-rag-cli:latest"
CONTAINER_NAME_PREFIX = "llm-rag-test-"

@pytest.fixture(scope="session")
def built_image():
    """
    Fixture to build the Docker image for llm_rag.
    """
    image = docker_image(
        name=IMAGE_TAG,
        path=os.path.join("..", "src", "llm_rag"),
        dockerfile="Dockerfile",
        pull=True,  # Ensures the base image is up-to-date
    )
    return image

@pytest.fixture
def container_instance(built_image):
    """
    Fixture to create and manage a Docker container instance from the built image.
    Automatically cleans up the container after the test.
    """
    container = DockerContainer(
        image=built_image,
        name=f"{CONTAINER_NAME_PREFIX}{built_image.id[:12]}",
        auto_remove=True,  # Automatically remove container after test
    )
    container.start()
    # Wait briefly to ensure the container is fully up
    time.sleep(2)
    yield container
    # Teardown is handled by auto_remove

def test_image_builds(built_image):
    """
    Test to ensure that the Docker image builds successfully.
    """
    assert built_image.id is not None, "Docker image failed to build."

def test_container_starts(container_instance):
    """
    Test that the container starts successfully with the built image.
    """
    status = container_instance.status
    assert status == "running", f"Container status is '{status}', expected 'running'."

def test_python_pipenv_installed(container_instance):
    """
    Test if Python 3.12 and Pipenv are installed and configured correctly in the container.
    """
    # Verify Python version
    exit_code, output = container_instance.exec_run("python --version")
    assert exit_code == 0, "Failed to execute 'python --version'."
    assert b"Python 3.12" in output, "Python 3.12 is not installed in the container."

    # Verify Pipenv installation
    exit_code, output = container_instance.exec_run("pipenv --version")
    assert exit_code == 0, "Failed to execute 'pipenv --version'."
    assert b"Pipenv" in output, "Pipenv is not installed in the container."

def test_container_runs_command(container_instance):
    """
    Test to verify that the container can execute a simple command correctly.
    """
    exit_code, output = container_instance.exec_run("echo 'Hello World'")
    assert exit_code == 0, "Failed to execute 'echo' command."
    assert output.strip() == b"Hello World", "Echo command did not return the expected output."

def test_entrypoint_execution(built_image):
    """
    Test that the container runs the entrypoint script correctly.
    """
    # Create a container without overriding the entrypoint
    entrypoint_container = DockerContainer(
        image=built_image,
        name=f"{CONTAINER_NAME_PREFIX}entrypoint-{built_image.id[:12]}",
        auto_remove=True,
    )
    entrypoint_container.start()
    
    # Wait for the entrypoint script to execute
    time.sleep(3)  # Adjust the sleep time based on the script's execution time

    # Retrieve container logs to verify entrypoint execution
    logs = entrypoint_container.logs()
    assert b"Entrypoint script executed" in logs, "Entrypoint script did not execute as expected."
