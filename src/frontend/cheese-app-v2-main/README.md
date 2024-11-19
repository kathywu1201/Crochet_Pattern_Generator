# Cheese App - APIs & Frontend

In this tutorial we will setup three containers:
* api-service
* frontend-simple
* frontend-react
* vector-db

## Prerequisites
* Have Docker installed

## Tutorial (23): Environment Setup
This tutorial covers setting up three main components:
1. A Python container for our API services
2. A web server container for the frontend
3. A vector database container for RAG (Retrieval Augmented Generation)

Each container will run in isolation but communicate with each other to create our complete cheese application.

### Clone the github repository
- Clone or download from [here](https://github.com/dlops-io/cheese-app-v2)

### Create a local **secrets** folder

It is important to note that we do not want any secure information in Git. So we will manage these files outside of the git folder. At the same level as the `cheese-app-v2` folder create a folder called **secrets**. Add the  secret file from the `ml-workflow` tutorial. 

Your folder structure should look like this:
```
   |-cheese-app-v2
     |-images
     |-src
       |---api-service
       |---frontend-simple
       |---frontend-react
       |---vector-db
   |-secrets
   |-persistent-folder
```

## Tutorial (23): Vector DB for RAG Setup
We will set up and initialize our vector database with cheese-related content for Retrieval Augmented Generation (RAG).

### Set up the Vector Database
1. Navigate to the vector-db directory:
```bash
cd cheese-app-v2/src/vector-db
```

2. Build and run the container:
```bash
sh docker-shell.sh
```

3. Initialize the database. Run this within the docker shell:
```bash
python cli.py --download --load --chunk_type recursive-split
```

This process will:
- Download the cheese knowledge base (chunks + embeddings)
- Load everything into the vector database for RAG functionality

> Note: This step is crucial for enabling our cheese assistant to provide accurate, knowledge-based responses.

Keep this container running while setting up the backend API service and frontend apps.

## Tutorial (23): Backend APIs
We will create a backend container running a FastAPI-based REST API service.

### Setup Steps

1. **Navigate to API Service Directory**
```bash
cd cheese-app-v2/src/api-service
```

2. **Build & Run Container**
```bash
sh docker-shell.sh
```

3. **Review Container Configuration** 
- Check `docker-shell.sh`: 
  - Port mapping: `-p 9000:9000`
  - Development mode: `-e DEV=1`
- Check `docker-entrypoint.sh`: Dev vs. Production settings

4. **Start the API Service**

Run the following command within the docker shell:
```bash
uvicorn_server
```
Verify service is running at `http://localhost:9000`

### Enable API Routes

1. **Enable All Routes in `api/service.py`**
```python
# Additional routers here
app.include_router(newsletter.router, prefix="/newsletters")
# app.include_router(podcast.router, prefix="/podcasts")
# app.include_router(llm_chat.router, prefix="/llm")
# app.include_router(llm_cnn_chat.router, prefix="/llm-cnn")
# app.include_router(llm_rag_chat.router, prefix="/llm-rag")
# app.include_router(llm_agent_chat.router, prefix="/llm-agent")
```
- Go to `http://localhost:9000/docs` and test the newsletters routes

- For each module we have a separate route:
  - Newsletters (`api/routers/newsletters.py`)
  - Podcasts (`api/routers/podcasts.py`)
  - LLM Chat (`api/routers/llm_chat.py`)
  - LLM + CNN Chat (`api/routers/llm_cnn_chat.py`)
  - LLM Rag (`api/routers/llm_rag_chat.py`)
  - LLM Agent  (`api/routers/llm_agent_chat.py`)

- Enable all the routes

### View API Docs
Fast API gives us an interactive API documentation and exploration tool for free.
- Go to `http://localhost:9000/docs`
- You can test APIs from this tool

Keep this container running while setting up the backend API service and frontend apps.
## Tutorial (24): Simple Frontend App
This section covers building a basic frontend using HTML & JavaScript that will interact with our API service.

### Setup Instructions
1. Navigate to the frontend directory:
```bash
cd cheese-app-v2/src/frontend-simple
```

2. Build & Run the container:
```bash
sh docker-shell.sh
```

3. Launch the development web server:
```bash
http-server
```


### Testing the Application

#### Home Page
- Visit `http://localhost:8080/index.html`
- You should see the cheese app landing page

#### Newsletter Feature
1. Open `http://localhost:8080/newsletters.html`
2. Verify that cheese newsletters are loading (requires running API service)
3. Review the code in `newsletters.html` to understand the API integration

#### Chat Feature
1. Open `http://localhost:8080/chat.html`
2. Test the chat by asking a cheese-related question (e.g., "How is cheese made?")
3. Review the code in `chat.html` to understand how the Gemini LLM integration works

> Note: The API service must be running for both the newsletter and chat features to work properly.

## Tutorial (25): React Frontend Setup

### Initial Setup
1. Navigate to the React frontend directory:
```bash
cd cheese-app-v2/src/frontend-react
```

2. Start the development container:
```bash
sh docker-shell.sh
```

### Dependencies Installation
First time only: Install the required Node packages
```bash
npm install
```

### Launch Development Server
1. Start the development server:
```bash
npm run dev
```

2. View your app at: http://localhost:3000

> Note: Make sure the API service container is running for full functionality
### Review App
- Go to Home page
- Go to Newsletters, Podcasts - Review functionality
- Go to Chat Assistant. Try LLM, LLM + CNN, RAG chats

### Review App Code
- Open folder `frontend-react/src`

### Data Services
- Data Service (`src/services/DataService.js`)
- Review Data Service methods that connects frontend to all backend APIs

### App Pages
- Open folder `frontend-react/src/app`
- Review the main app pages
  - Home (`src/app/page.jsx`)
  - Newsletters (`src/app/newsletters/page.jsx`)
  - Podcasts (`src/app/podcasts/page.jsx`)
  - Chat Assistant (`src/app/chat/page.jsx`)

### App Components
- Open folder `frontend-react/src/components`
- Review components of the app
  - Layout for common components such as Header, Footer
  - Chat for all the chat components


---
## Docker Cleanup

### Make sure we do not have any running containers and clear up an unused images
* Run `docker container ls`
* Stop any container that is running
* Run `docker system prune`
* Run `docker image ls`
