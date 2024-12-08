# AC215 YarnMaster!!


#### Project Organization

```
├── Readme.md
├── data 
├── notebooks
│   └── eda.ipynb
├── references/
├── reports
│   └── APCOMP215 Proposal.pdf
├── tests
│   ├── documentations.txt
│   ├── test_ImageDescription.py
│   ├── test_ImageVector.py
│   ├── test_integration.py
│   └── test_pdfProcessor.py
└── src
    ├── data_gathering
    │   ├── Pipfile
    │   ├── Pipfile.lock
    │   ├── docker-shell.sh
    │   ├── docker-entrypoint.sh
    │   ├── cli.py
    │   ├── data_scraping.py
    │   ├── data_upload.py
    ├── image_descriptions
    │   ├── Pipfile
    │   ├── Pipfile.lock
    │   ├── docker-shell.sh
    │   ├── docker-entrypoint.sh
    │   ├── Dockerfile
    │   ├── echo
    │   ├── cli.py
    ├── pdf_processor
    │   ├── Pipfile
    │   ├── Pipfile.lock
    │   ├── docker-shell.sh
    │   ├── docker-entrypoint.sh
    │   ├── Dockerfile·
    │   ├── cli.py
    ├── image_2_vector
    │   ├── Pipfile
    │   ├── Pipfile.lock
    │   ├── docker-shell.sh
    │   ├── docker-entrypoint.sh
    │   ├── Dockerfile
    │   ├── cli.py
    ├── llm_rag
    │   ├── Pipfile
    │   ├── Pipfile.lock
    │   ├── docker-shell.sh
    │   ├── docker-entrypoint.sh
    │   ├── docker-shell.sh
    │   ├── docker-compose.yml
    │   ├── rag.py
    │   ├── data_preparation.py
    ├── text-generation
    │   └── llm-finetuning
    │       ├── Pipfile
    │       ├── Pipfile.lock
    │       ├── docker-shell.sh
    │       ├── docker-entrypoint.sh
    │       ├── Dockerfile
    │       └── processing_pdfs.py
    ├── api-service
    │   ├── Pipfile
    │   ├── Pipfile.lock
    │   ├── docker-shell.sh
    │   ├── docker-entrypoint.sh
    │   ├── Dockerfile
    │   ├── api
    │   │   ├── service.py
    │   │   ├── routers
    │   │   │   ├── llm_rag_chat.py
    │   │   │   ├── llm_agent_chat.py
    │   │   └── utils
    │   │       ├── chat_utils.py
    │   │       ├── llm_rag_utils.py
    │   │       ├── llm_agent_utils.py
    │   │       └── llm_image_utils.py
    ├── vectorDB
    │   ├── Pipfile       
    │   ├── Pipfile.lock           
    │   ├── Dockerfile             
    │   ├── docker-shell.sh         
    │   ├── docker-entrypoint.sh    
    │   ├── docker-compose.yml      
    │   ├── cli.py                  
    │   ├── semantic_splitter.py  
    ├── frontend_react
    │   ├── public/assets
    │   │   └── *.png
    │   └── src
    │       ├── components/
    │       ├── app/
    │       └── services/
 
```

# AC215 - Final Project - YarnMaster!!

**Team Members**
Shiqi Wang, Yanfeiyun Wu, Jiahui Zhang, Wanying Zhang

**Group Name**
YarnMaster

**Project**
In this project, we aim to develop a Deep Learning model capable of interpreting images of crochet products and generating corresponding, detailed pattern instructions. Despite crochet's popularity as a creative and therapeutic pastime, enthusiasts like us often struggle to find clear, step-by-step instructions for intriguing designs we encounter online or in person due to scarce resources. Existing tools can transform written instructions into visual 3D models, but there is a significant gap in generating pattern instructions directly from images of finished products. To address this need, we envision not only creating a model that provides meaningful instructions but also developing an app tailored for crochet enthusiasts. This platform would allow users to upload pictures of crochet items they find interesting, receive detailed patterns, and foster an online community where they can connect, share their creations, and inspire each other [project proposal](./reports/APCOMP215%20Proposal.pdf).

## Previous Milestones

Below are the previous milestones we've done.

- Milestone 2:
    - Data scraping from the website, data preprocessing, language model finetune and RAG. 
    - Each component are in separte contaienrs.
- Miletone 3:
    - Seprate the data preprocessing container to `pdf_processor` and `image_2_vector`.
- Milestone 4:
    - Setup frontend interface, backend API, tests for different functionalities.
    - Additional `fontend-react` and `api-service` containers.

## Final Deliverables

### Overview of YarnMaster
[add the project description]

#### How to use YarnMaster? ####




### Deployment Instructions

#### App Deployment to GCP (Ansible) ####

The deployment of our application on a Google Cloud Platform (GCP) Virtual Machine (VM) is fully automated using Ansible. The process involves configuring secrets, building Docker images, setting up the VM, and deploying containers with a web server. Below is a step-by-step guide with explanations.

1. **Configuration**
   - Begin by updating the `inventory.yml` file with the following details:
     - Ansible user credentials.
     - GCP project information.
     - GCP VM instance details.
   - Ensure necessary secrets, such as SSH keys and GCP credentials, are properly set up for secure deployment.

2. **Build and Push Docker Containers**
   - Build Docker images for the application components: `vector-db`, `api-service`, and `frontend-react`.  
     - Run the command: `ansible-playbook deploy-docker-images.yml -i inventory.yml`.  
   - Push these images to the Google Container Registry (GCR).  
   - Tag all images with a timestamp to maintain consistency between components.  
   - Note: This step may take time, especially for large images like the `api-service`.

3. **Create Compute Instance**
   - Create a GCP VM configured with a mounted persistent disk for data storage.  
   - Allow HTTP traffic on port 80 and HTTPS traffic on port 443 to enable web access.  
   - Use the command: `ansible-playbook deploy-create-instance.yml -i inventory.yml --extra-vars cluster_state=present`.  
   - Once the VM is created, update the `inventory.yml` file with the compute instance's external IP address.

4. **Provision Compute Instance**
   - Set up the VM with required dependencies and configurations.  
   - This includes installing Docker, setting up the Docker network, and preparing the environment for container deployment.  
   - Use the command: `ansible-playbook deploy-provision-instance.yml -i inventory.yml`.

5. **Deploy Docker Containers**
   - Pull the pre-built Docker images from GCR and run them on the VM:
     - **Vector-DB:** Runs on port 8000.  
     - **API Service:** Runs on port 9000.  
     - **Frontend:** Runs on port 3000.  
   - Secrets (e.g., GCP credentials) are mounted into the containers to ensure secure access to necessary resources.  
   - Deploy the containers using: `ansible-playbook deploy-setup-containers.yml -i inventory.yml`.

6. **Webserver Configuration**
   - Set up Nginx as a reverse proxy to route incoming HTTP traffic between the `frontend` and `API service`:  
     - Configure `nginx.conf` with routing rules.  
     - Deploy the web server using: `ansible-playbook deploy-setup-webserver.yml -i inventory.yml`.  
   - This creates a new Nginx container to manage traffic on port 80.

7. **Access the Application**
   - Use the VM's external IP address to access the application in a web browser at: `http://<External IP>/`.

By using Ansible for automation, this deployment process is efficient and repeatable. It handles everything from building Docker images to configuring the VM and deploying containers, ensuring a streamlined setup. The application is fully accessible via the VM’s external IP, with Nginx managing traffic between components.

#### App Deployment using Kubernetes ####

This guide outlines the process for deploying the Crochet App on a Kubernetes cluster using Ansible. The deployment leverages the `deploy-k8s-cluster.yml` playbook and the `inventory.yml` configuration file for automation.

**1. Build and Push Docker Containers (Optional)**  
   - If the Docker containers for the application are not already built and pushed to the Google Container Registry (GCR), execute:  
     `ansible-playbook deploy-docker-images.yml -i inventory.yml`
   - This step builds Docker images for the frontend, API service, and ChromaDB components and pushes them to GCR, tagging the images for version control.

**2. Create and Deploy Kubernetes Cluster**  
   - Use the following command to create and configure a Google Kubernetes Engine (GKE) cluster:  
     `ansible-playbook deploy-k8s-cluster.yml -i inventory.yml --extra-vars cluster_state=present`
   - This process includes:
     1. **Cluster Creation:** Creates a GKE cluster (`crochet-app-cluster`) with autoscaling (1–2 nodes) and nodes of type `n2d-standard-2` with 30 GB disk space.
     2. **Namespace Setup:** Creates the `crochet-app-cluster-namespace` for managing Kubernetes resources.
     3. **Ingress Controller:** Installs Nginx ingress using Helm for external access to cluster services.
     4. **Persistent Volume Claims:** Sets up PVCs for app data storage and ChromaDB storage.
     5. **Secrets Management:** Imports GCP credentials into the cluster as Kubernetes secrets for secure access.
     6. **Application Deployment:** Deploys:
        - **Frontend Service** (exposed on port 3000).  
        - **API Service** (exposed on port 9000 with GCP credentials and persistent storage).  
        - **ChromaDB Service** (exposed on port 8000 with persistent storage).
     7. **Data Initialization:** Executes a Kubernetes job to load initial data into ChromaDB.
     8. **Service Exposure:** Creates NodePort services for frontend, API, and ChromaDB components.
     9. **Ingress Configuration:** Sets up routing to:
        - `/` for frontend traffic.  
        - `/api/` for API service traffic.

**3. Access the Application**  
   - After successful deployment, retrieve the ingress IP from the terminal output.
   - Open the app in a web browser using:  
     ```http://<YOUR INGRESS IP>.sslip.io```http://34.74.248.170.sslip.io

This deployment process automates the setup of a Kubernetes cluster, the configuration of resources, and the deployment of services. It ensures scalability, security, and efficient resource management for the Crochet App, making it accessible via the provided ingress IP.

### Prerequisites

Our app is fully implemented using containers, which mean it is necessary to have access to the Docker Container. To run the Docker container locally, ensure that Docker is installed and running on your system. You should also have access to the required secrets, such as GCP credentials and SSH keys, properly set up in the `/secrets` directory. Verify that you have configured the `inventory.yml` file with accurate project details, including the GCP project ID and authentication settings. Additionally, confirm that your system has access to the internet to pull necessary images and authenticate with GCP. Finally, check that your Docker environment has sufficient resources (CPU, memory, and disk space) to run the container smoothly.

### Setup Instructions

This is the instruction for setting up the deployment process for the first time.

1. **Setup Docker Container**  
   - Use Docker to create a deployment container with all required tools.  
   - Update `docker-shell.sh` with your GCP project ID and run `sh docker-shell.sh`.  
   - Verify tools with `gcloud`, `ansible`, and `kubectl` commands.  
   - Authenticate to GCP with `gcloud auth list`.

2. **SSH Setup**  
   - Enable OS Login: `gcloud compute project-info add-metadata --project <YOUR GCP_PROJECT> --metadata enable-oslogin=TRUE`.  
   - Generate SSH key: `ssh-keygen -f /secrets/ssh-key-deployment`.  
   - Add public SSH key to GCP: `gcloud compute os-login ssh-keys add --key-file=/secrets/ssh-key-deployment.pub`.  
   - Note the generated username for later use.

