# This file follows the same logic as in gemini_finetuning.py in gemini_finetuning folder, but added the
# manually validation check component to manually check the quality of the output before deploy the model

from kfp import dsl

@dsl.component(
    base_image="python:3.10", 
    packages_to_install=["google-cloud-aiplatform", "vertexai"]
)
def model_training(
    project: str = "",
    location: str = "",
    train_dataset: str = "",
    validation_dataset: str = "",
    source_model: str = "",
    display_name: str = "",
    output_bucket: str = "",
):
    import os
    import time
    from vertexai.preview.tuning import sft
    from google.cloud import storage

    # Initialize Vertex AI
    from vertexai import init
    init(project=project, location=location)

    # Fine-tune the model
    print("Starting fine-tuning...")
    sft_tuning_job = sft.train(
        source_model=source_model,
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        epochs=1,
        adapter_size=4,
        learning_rate_multiplier=1.0,
        tuned_model_display_name=display_name,
    )

    print("Fine-tuning job started...")
    time.sleep(60)  # Allow time for the job to start
    sft_tuning_job.refresh()

    while not sft_tuning_job.has_ended:
        time.sleep(60)
        sft_tuning_job.refresh()
        print("Fine-tuning in progress...")

    print(f"Tuned model name: {sft_tuning_job.tuned_model_name}")
    print(f"Tuned model endpoint name: {sft_tuning_job.tuned_model_endpoint_name}")
    print(f"Experiment details: {sft_tuning_job.experiment}")

    # Save the tuned model details for validation
    client = storage.Client()
    bucket = client.bucket(output_bucket)
    blob = bucket.blob("fine_tuned_model_info.txt")
    blob.upload_from_string(f"Tuned Model Name: {sft_tuning_job.tuned_model_name}\n")


# Define a component for generating sample outputs
@dsl.component(base_image="python:3.10")
def generate_sample_outputs(
    project: str = "",
    location: str = "",
    tuned_model_name: str = "",
    output_bucket: str = "",
    sample_prompts: list = [],
):
    from google.cloud import storage
    from vertexai.generative_models import GenerativeModel
    from vertexai import init

    # Initialize Vertex AI
    init(project=project, location=location)

    # Load the fine-tuned model
    model = GenerativeModel(tuned_model_name)
    samples = {}

    for prompt in sample_prompts:
        samples[prompt] = model.generate(prompt).text

    # Save outputs to GCS
    client = storage.Client()
    bucket = client.bucket(output_bucket)
    for prompt, output in samples.items():
        blob = bucket.blob(f"qualitative_outputs/{prompt}.txt")
        blob.upload_from_string(output)


# Define a validation step for manual review of sample outputs
@dsl.component(base_image="python:3.10")
def manual_review_step(output_bucket: str):
    from google.cloud import storage

    client = storage.Client()
    bucket = client.bucket(output_bucket)
    blobs = list(bucket.list_blobs(prefix="qualitative_outputs/"))

    if not blobs:
        raise ValueError("No sample outputs found for review.")

    print("Review the following sample outputs in the bucket:")
    for blob in blobs:
        print(f" - {blob.name}")

    approval = input("Do you approve the model for deployment? (yes/no): ").strip().lower()
    if approval != "yes":
        raise ValueError("Qualitative assessment failed. Deployment halted.")


# Define a deployment component
@dsl.component(
    base_image="python:3.10", 
    packages_to_install=["google-cloud-aiplatform"]
)
def model_deploy(
    project: str = "",
    bucket_name: str = "",
    display_name: str = "",
):
    print("Starting Model Deployment...")

    from google.cloud import aiplatform as aip

    # Initialize Vertex AI SDK for Python
    aip.init(project=project)

    # Model details
    artifact_uri = f"gs://{bucket_name}/model"
    serving_container_image_uri = "us-docker.pkg.dev/vertex-ai/prediction/tf2-cpu.2-12:latest"

    # Upload and deploy the model
    print(f"Uploading model from {artifact_uri}...")
    model = aip.Model.upload(
        display_name=display_name,
        artifact_uri=artifact_uri,
        serving_container_image_uri=serving_container_image_uri,
    )

    print("Deploying model to an endpoint...")
    endpoint = model.deploy(
        deployed_model_display_name=display_name,
        machine_type="n1-standard-4",
        min_replica_count=1,
        max_replica_count=1,
        traffic_split={"0": 100},
    )

    print(f"Model deployed successfully to endpoint: {endpoint.resource_name}")

