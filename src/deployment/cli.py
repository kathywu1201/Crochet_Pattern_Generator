"""
Module that contains the command line app.

Typical usage example from command line:
        python cli.py
"""

import os
import argparse
import random
import string
from kfp import dsl
from kfp import compiler
import google.cloud.aiplatform as aip
from model import (
    model_training as model_training_job,
    model_deploy as model_deploy_job,
    generate_sample_outputs,
    manual_review_step,
)

GCP_PROJECT = os.environ["GCP_PROJECT"]
GCS_BUCKET_NAME = os.environ["GCS_BUCKET_NAME"]
BUCKET_URI = f"gs://{GCS_BUCKET_NAME}"
PIPELINE_ROOT = f"{BUCKET_URI}/pipeline_root/root"
GCS_SERVICE_ACCOUNT = os.environ["GCS_SERVICE_ACCOUNT"]
GCS_PACKAGE_URI = os.environ["GCS_PACKAGE_URI"]
GCP_REGION = os.environ["GCP_REGION"]

# Read the docker tag file
with open(".docker-tag-ml") as f:
    tag = f.read()

tag = tag.strip()

print("Tag>>", tag, "<<")

PDF_PROCESSOR_IMAGE = f"gcr.io/{GCP_PROJECT}/crochet-app-pdf-processor:{tag}"
IMAGE_DESCRIPTIONS_IMAGE = f"gcr.io/{GCP_PROJECT}/crochet-app-image-descriptions:{tag}"


def generate_uuid(length: int = 8) -> str:
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=length))


def pdf_processor():
    print("pdf_processor()")

    # Define a Container Component
    @dsl.container_component
    def pdf_processor():
        container_spec = dsl.ContainerSpec(
            image=PDF_PROCESSOR_IMAGE,
            command=[],
            args=[
                "cli.py",
                "--download",
                "--process",
                "--upload",
                "--folders socks+pillows+rugs",
                f"--bucket {GCS_BUCKET_NAME}",
            ],
        )
        return container_spec

    # Define a Pipeline
    @dsl.pipeline
    def pdf_processor_pipeline():
        pdf_processor()

    # Build yaml file for pipeline
    compiler.Compiler().compile(
        pdf_processor_pipeline, package_path="pdf_processor.yaml"
    )

    # Submit job to Vertex AI
    aip.init(project=GCP_PROJECT, staging_bucket=BUCKET_URI)

    job_id = generate_uuid()
    DISPLAY_NAME = "crochet-app-pdf-processor-" + job_id
    job = aip.PipelineJob(
        display_name=DISPLAY_NAME,
        template_path="pdf_processor.yaml",
        pipeline_root=PIPELINE_ROOT,
        enable_caching=False,
    )

    job.run(service_account=GCS_SERVICE_ACCOUNT)


def image_descriptions():
    print("image_descriptions()")

    # Define a Container Component for Image Descriptions
    @dsl.container_component
    def image_descriptions():
        container_spec = dsl.ContainerSpec(
            image=IMAGE_DESCRIPTIONS_IMAGE,
            command=[],
            args=[
                "cli.py",
                "--download",
                "--clean_instructions",
                "--process",
                "--split",
                "--upload",
                f"--bucket {GCS_BUCKET_NAME}",
            ],
        )
        return container_spec

    # Define a Pipeline
    @dsl.pipeline
    def image_descriptions_pipeline():
        image_descriptions()

    # Build yaml file for pipeline
    compiler.Compiler().compile(
        image_descriptions_pipeline, package_path="image_descriptions.yaml"
    )

    # Submit job to Vertex AI
    aip.init(project=GCP_PROJECT, staging_bucket=BUCKET_URI)

    job_id = generate_uuid()
    DISPLAY_NAME = "crochet-app-image-descriptions-" + job_id
    job = aip.PipelineJob(
        display_name=DISPLAY_NAME,
        template_path="image_descriptions.yaml",
        pipeline_root=PIPELINE_ROOT,
        enable_caching=False,
    )

    job.run(service_account=GCS_SERVICE_ACCOUNT)


def model_training():
    print("model_training()")

    # Define a Pipeline
    @dsl.pipeline
    def model_training_pipeline():
        model_training_job(
            project=GCP_PROJECT,
            location=GCP_REGION,
            train_dataset="gs://{GCS_BUCKET_NAME}train.jsonl",
            validation_dataset="gs://crochet-patterns/validation.jsonl",
            source_model="gemini-1.5-flash-002",
            display_name="finetuned-gemini-v2",
            output_bucket=GCS_BUCKET_NAME,
        )

    # Build yaml file for pipeline
    compiler.Compiler().compile(
        model_training_pipeline, package_path="model_training.yaml"
    )

    # Submit job to Vertex AI
    aip.init(project=GCP_PROJECT, staging_bucket=BUCKET_URI)

    job_id = generate_uuid()
    DISPLAY_NAME = "crochet-app-model-training-" + job_id
    job = aip.PipelineJob(
        display_name=DISPLAY_NAME,
        template_path="model_training.yaml",
        pipeline_root=PIPELINE_ROOT,
        enable_caching=False,
    )

    job.run(service_account=GCS_SERVICE_ACCOUNT)


def model_deploy():
    print("model_deploy()")

    # Define a Pipeline
    @dsl.pipeline
    def model_deploy_pipeline():
        model_deploy_job(
            project=GCP_PROJECT,
            bucket_name=GCS_BUCKET_NAME,
            display_name="finetuned-gemini-v2",
        )

    # Build yaml file for pipeline
    compiler.Compiler().compile(
        model_deploy_pipeline, package_path="model_deploy.yaml"
    )

    # Submit job to Vertex AI
    aip.init(project=GCP_PROJECT, staging_bucket=BUCKET_URI)

    job_id = generate_uuid()
    DISPLAY_NAME = "crochet-app-model-deploy-" + job_id
    job = aip.PipelineJob(
        display_name=DISPLAY_NAME,
        template_path="model_deploy.yaml",
        pipeline_root=PIPELINE_ROOT,
        enable_caching=False,
    )

    job.run(service_account=GCS_SERVICE_ACCOUNT)


def pipeline():
    print("pipeline()")

    @dsl.container_component
    def pdf_processor():
        container_spec = dsl.ContainerSpec(
            image=PDF_PROCESSOR_IMAGE,
            command=[],
            args=[
                "cli.py",
                "--download",
                "--process",
                "--upload",
                "--folders socks+pillows+rugs",
                f"--bucket {GCS_BUCKET_NAME}",
            ],
        )
        return container_spec

    @dsl.container_component
    def image_descriptions():
        container_spec = dsl.ContainerSpec(
            image=IMAGE_DESCRIPTIONS_IMAGE,
            command=[],
            args=[
                "cli.py",
                "--download",
                "--clean_instructions",
                "--process",
                "--split",
                "--upload",
                f"--bucket {GCS_BUCKET_NAME}",
            ],
        )
        return container_spec
    
    # Define a Pipeline
    @dsl.pipeline
    def ml_pipeline():
        # PDF PROCESSOR
        pdf_processor_task = (
            pdf_processor()
            .set_display_name("PDF PROCESSOR")
            .set_cpu_limit("500m")
            .set_memory_limit("2G")
        )
        # Image Descriptions
        image_descriptions_task = (
            image_descriptions()
            .set_display_name("Image Descriptions")
            .after(pdf_processor_task)
        )
        # Model Training
        model_training_task = (
            model_training_job(
                project=GCP_PROJECT,
                location=GCP_REGION,
                train_dataset="gs://{GCS_BUCKET_NAME}/training/image_descriptions_jsonl/train.jsonl",
                validation_dataset="gs://{GCS_BUCKET_NAME}/training/image_descriptions_jsonl/validation.jsonl",
                source_model="gemini-1.5-flash-002",
                display_name="finetuned-gemini-v2",
                output_bucket=GCS_BUCKET_NAME,
            )
            .set_display_name("Model Training")
            .after(image_descriptions_task)
        )
        # Generate Sample Outputs
        sample_outputs_task = (
            generate_sample_outputs(
                project=GCP_PROJECT,
                location=GCP_REGION,
                tuned_model_name="finetuned-gemini-v2",
                output_bucket=GCS_BUCKET_NAME,
                sample_prompts=[
                    "Create a heart-shaped coaster crochet pattern",
                    "Write instructions for a crochet scarf",
                    "Design a crochet flower pattern",
                ],
            )
            .set_display_name("Generate Sample Outputs")
            .after(model_training_task)
        )
        # Manual Review
        manual_approval = manual_review_step(
            output_bucket=GCS_BUCKET_NAME,
        ).after(sample_outputs_task)
        # Model Deployment
        model_deploy_task = (
            model_deploy_job(
                project=GCP_PROJECT,
                bucket_name=GCS_BUCKET_NAME,
                display_name="finetuned-gemini-v2",
            )
            .set_display_name("Model Deploy")
            .after(manual_approval)
        )

    # Build yaml file for pipeline
    compiler.Compiler().compile(ml_pipeline, package_path="pipeline.yaml")

    # Submit job to Vertex AI
    aip.init(project=GCP_PROJECT, staging_bucket=BUCKET_URI)

    job_id = generate_uuid()
    DISPLAY_NAME = "crochet-app-pipeline-" + job_id
    job = aip.PipelineJob(
        display_name=DISPLAY_NAME,
        template_path="pipeline.yaml",
        pipeline_root=PIPELINE_ROOT,
        enable_caching=False,
    )

    job.run(service_account=GCS_SERVICE_ACCOUNT)


def main(args=None):
    print("CLI Arguments:", args)

    if args.pdf_processor:
        pdf_processor() # Run just PDF Processor

    if args.image_descriptions:
        print("Image Descriptions")
        image_descriptions() # Run just Image Descriptions

    if args.model_training:
        print("Model Training")
        model_training() # Run just Model Training

    if args.model_deploy:
        print("Model Deploy")
        model_deploy() # Run just Model Deploy

    if args.pipeline:
        pipeline() # Run the entire pipeline, including qualitative validation


if __name__ == "__main__":
    # Generate the inputs arguments parser
    parser = argparse.ArgumentParser(description="Workflow CLI")

    parser.add_argument(
        "--pdf_processor",
        action="store_true",
        help="Run just the PDF PROCESSOR",
    )
    parser.add_argument(
        "--image_descriptions",
        action="store_true",
        help="Run just the Image Descriptions",
    )
    parser.add_argument(
        "--model_training",
        action="store_true",
        help="Run just Model Training",
    )
    parser.add_argument(
        "--model_deploy",
        action="store_true",
        help="Run just Model Deployment",
    )
    parser.add_argument(
        "--pipeline",
        action="store_true",
        help="Crochet App Pipeline",
    )

    args = parser.parse_args()

    main(args)

