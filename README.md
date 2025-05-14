# Digit Classifier

Application for Machine Learning Institute program. 

Objective: Build a MNIST digit Classifier. See https://programme.mlx.institute/interview/project

## Dev - running the app locally

### Prerequisites
Needed:
- Python (v3.9.6)
- Poetry for python package management (`brew install poetry` or see https://python-poetry.org/ to install. v2.1.3 or higher)
- [colima](https://github.com/abiosoft/colima) for using docker without needing Docker Desktop (`brew install colima`) 
    - For colima to work, install docker (`brew install docker`) 

### 1. Initial python set up
To run the digit_classifier python files:
1. Ensure poetry is using python v3.9.6 (see commands listed: https://python-poetry.org/docs/managing-environments/ - e.g. using `poetry env use 3.9`)
2. `poetry env activate` to use the poetry virtual environment created
    - (To deactivate virtual env if needed, run `deactivate`)
3. `poetry install` to install project requirements

### 2. PostgreSQL database
#### To set up and start the docker container
1. `colima start` to start up docker
    - To stop colima and the VM, run `colima stop`
2. Run `docker pull postgres` to get a PostgreSQL Docker image
3. Run `docker run --name postgres_container -e POSTGRES_PASSWORD=<POSTGRES_PASSWORD> -d -p <DB_PORT>:<DB_PORT> -v postgres_data:/var/lib/postgresql/data postgres` to run the PostgreSQL container
- Get the `.env` file from one of the Dev's (the only dev - Helen ;D) to get the `<POSTGRES_PASSWORD>` and `<DB_PORT>` values
- To verify the docker is up and running, run `docker ps`
This sets up the database within a docker container, and the frontend app will interact with using the [psycopg2 package](https://www.psycopg.org/docs/install.html#build-prerequisites)

#### To restart docker container
If you have previously run the above set up steps (you can verify that it Exited by running `docker ps -a` and seeing the docker container with the name `postgres_container`), you can restart the container by running `docker restart postgres_container`.
- To verify it is up and running, run `docker ps` and view status of `postgres_container`

### 3. Streamlit Front end
To run front end, ensure the postgresSQL database docker container is up and running. Then run the script: `streamlit run src/digit_classifier/app.py` and it will create a localhost URL to view. 

---

## Tasks
For full details, see: https://programme.mlx.institute/interview/project

Live example of the application: https://mnist-example.mlx.institute

1. ✅ **Train a PyTorch Model**
    - ✅ Develop a basic PyTorch model to classify handwritten digits from the MNIST dataset.
    - ✅ Train it locally and confirm that it achieves a reasonable accuracy.
2. ✅ **Interactive Front-End**
    - ✅ Create a web interface (using Streamlit) where users can draw a digit on a canvas or input area.
    - ✅ When the user submits the drawing, the web app should run the trained PyTorch model to produce:
        - ✅ Prediction: the model's guess at the digit (0–9).
        - ✅ Confidence: the model's probability for its prediction.
        - ✅ True Label: allow the user to manually input the correct digit so you can gather feedback.
3. ✅ **Logging with PostgreSQL**
- ✅ Every time a prediction is made, log these details to a PostgreSQL database:
    - ✅ Timestamp
    - ✅ Predicted digit
    - ✅ User-provided true label
4. **Containerization with Docker**
- Use Docker to containerize:
    - The PyTorch model/service
    - The Streamlit web app
    - The PostgreSQL database
- Use Docker Compose to define your multi-container setup in a docker-compose.yml file.
5. **Deployment**
- Set up a self-managed server (e.g., Hetzner's basic instance) or any other environment where you can install Docker and control the deployment end-to-end.
- Deploy your containerized application to the server and make it accessible via a public IP or domain.
6. **Add project to GitHub**
- Add your project to GitHub.
- Make sure to include a README with a link to the live application.
- Share the link to your GitHub repository with us via the application form.

## Resources
- Tutorial used to learn Machine learning on pytorch: # Tutorial to learn pytorch adapted from https://www.learnpytorch.io/03_pytorch_computer_vision/ 