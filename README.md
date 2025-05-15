# Digit Classifier

Objective: Build a MNIST digit Classifier (as part of the application for the Machine Learning Institute program.). See https://programme.mlx.institute/interview/project

⚠️ **DISCLAIMER**: This codebase is a Proof of Concept, completed over 4 days (see daily log below)! It is not production level code or deployment. Read it at your own risk 😉 ⚠️ 

## See it in action
Public URL to live application: http://ec2-3-10-138-208.eu-west-2.compute.amazonaws.com:8500/

👉 NOTE! The public URL isn't always up and running since AWS EC2 instances aren't _that_ cheap (especially over time!). This above link will be live during working hours (9am until 5pm on weekdays), until Tuesday 20th May 2025. Contact me if after this point you need it live. Below are some screenshots to evidence it all worked anyway:

- Where an AWS EC2 instance is up and running, with the public IP address highlighted:

![EC2 instance](https://github.com/user-attachments/assets/81802fb0-0840-4f50-a4c0-7b58eebf483c)

- Going to the port 8500 of the public IP:

![Access](https://github.com/user-attachments/assets/4db00d4f-42f9-432b-b7de-185767c37bf5)

- See how it updates the postgreSQL database with an entry:

![Update](https://github.com/user-attachments/assets/e5480dbb-a8ee-4c72-8a54-274a9dab7fe0)

## Architecture
- Python codebase, split into the different apps (the frontend, database and model). Code is under src/
    - **frontend**: generates the frontend where the user can draw a 0-9 digit, and see what the model predicted digit is with confidence level. User can input feedback (the 'true' digit), and see all feedback records
    - **Database service/API**: used to store the feedback records
    - **Model service/API**: has the machine learning model, trained and tested on the MNIST dataset.
- poetry as the package manager. The packages are split into what is required for each app:
    - **frontend**: streamlit used
    - **Database service/API**: uses a postgreSQL database, and fastapi + uvicorn for the API interface that the frontend can interact with
    - **Model service/API**: uses pytorch to train and test a model. fastapi + uvicorn for the API interface that the frontend can interact with
- Dockerfiles for each app to build and deploy, and docker compose for a multi-container setup
- AWS (with EC2 instance launced) used to host and allow public IP access

## Dev - running the app locally

### Prerequisites
Needed:
- Python (v3.9.6)
- Poetry for python package management (`brew install poetry` or see https://python-poetry.org/ to install. v2.1.3 or higher)
- [colima](https://github.com/abiosoft/colima) for using docker without needing Docker Desktop (`brew install colima`) 
    - For colima to work, install docker (`brew install docker`) 
- .env file needs to be populated correctly (get this from Helen). Example contents:
    ```python
    POSTGRES_USERNAME=xxx
    POSTGRES_PASSWORD=xxx
    DB_HOST=xxx
    DB_PORT=xxx
    MODEL_API_URL=xxx
    DATABASE_API_URL=xxx
    ```

Skip down to the docker-compose instructions for the most efficient way to get it up and running locally

### 1. Initial python set up
To run the digit_classifier python files:
1. Ensure poetry is using python v3.9.6 (see commands listed: https://python-poetry.org/docs/managing-environments/ - e.g. using `poetry env use 3.9`)
2. `poetry env activate` to use the poetry virtual environment created
    - (To deactivate virtual env if needed, run `deactivate`)
3. `poetry install` to install project requirements. The packages are split depending on which service you need running. To install frontend/database/model dependencies, run (delete as necessary )`poetry install --with frontend/database/model`, 

### 2. Model service/API
Code for the model service/API can be found in `src/digit_classifier/model`
The model has been trained on the MNIST dataset. The code ensures that the model is usable by ensuring the model loss is < 0.5 and model accuracy is > 90% when testing on the MNIST testing dataset.
- To load the service locally, use `uvicorn digit_classifier.model.api:app --reload`.
    - This runs the backend on port 8000, to check it is up and running go to: http://localhost:8000/healthcheck to see a response.

### 3. PostgreSQL database
Code for the postgreSQL database set up and SQL queries can be found in `src/digit_classifier/database`
A postgreSQL database is used to log the feedback of the user, along with the prediction and confidence level. All entries to the database is displayed to the user on the frontend.

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

#### To run the database API
Run `uvicorn digit_classifier.database.api:appdb --reload --host 0.0.0.0 --port 8001`, which will run it on localhost port 8001. Check http://localhost:8001/healthcheck for a response to see it up and running

### 4. Streamlit Front end
Code for the streamlit front end can be found in `src/digit_classifier/frontend`
To run front end locally, ensure the following are running:
- postgresSQL database docker container
- database API
- model service API
Then run the script: `streamlit run src/digit_classifier/frontend/app.py` and it will create a localhost URL to view. 

## Docker compose
In order to bypass the above steps and just run several docker containers at once (namely the postgreSQL database docker, database API docker, model API docker and frontend docker), use docker-compose.
1. Ensure docker-compose has been installed (`brew install docker-compose`). Version 2.36.0 at least is needed.
2. Then use `docker composer up` to get it up and running. See the entire end to end application on http://0.0.0.0:8500
    - If you are working on the code, in order to rebuild the images, run `docker compose down` then `docker-compose build <name of image to rebuild>`

### To set up and start the docker container
Starting individual docker containers is a bit redundant since they rely on connections to other docker containers. However leaving this here just for reference.
1. To build the docker image named 'frontend/database/model' (delete as necessary) run `docker build --file Dockerfile.<frontend/database/model> -t <frontend/database/model> .`
2. To run the built image, run `docker run -p <port number>:<port number> <frontend/database/model>`
You should be able to see the app running here: http://127.0.0.1:<port number>


## Deployment
### Prerequistes
- For AWS CLI access: need aws-vault - setup using [instructions](https://github.com/99designs/aws-vault/tree/master) and awscli for using AWS CLI tools
    - For me, run `aws-vault exec personal -- aws s3 ls` to check access
- An AWS account set up and role allowing for EC2 instance launches.

### How to deploy
Ideally I'd have loved to have used terraform, but for the interest of time (Proof of Concept remember!) I used AWS console.
Add tags to all AWS resources produced for good practice.
1. Generate a new key pair. Log into your AWS console and navigate to EC2 > Key pairs then:
- Generate a new Key pair with a custom name.
- Use type RSA and .pem file format.
- Save the .pem file (for ease of use I saved it in the project repo, .gitignore will make sure that file isn't committed)
- Set file permissions of the .pem file using `chmod 400 <file name>.pem`
2. In your AWS console, launch an EC2 instance. Select:
- Amazon Linux 2023 AMI
- Instance type r5.large (the only other instance type I tried was t2.micro but Error code 137 i.e. not enough memory plagued me)
- Select the key pair created in the previous step
- Storage: I increased to 15GiB gp3 (this may be overkill but haven't experimented with reducing it yet)
- Security group: either select an existing, or create a new one and ensure only your IP address can be used to SSH into instance
3. Once EC2 instance has launched, note the public IP address (a series of numbers and dots)
4. Run the following commands (replace the `<...>` where necessary) in order to SSH into the EC2 instance, install docker + docker-compose and deploy the apps
```console
scp -r -i <name of pem file>.pem <location from ~ of code> ec2-user@<public IP address>:~/
ssh -i <name of pem file>.pem ec2-user@<public IP address>
sudo yum update
sudo yum install docker # then hit y
sudo curl -L https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m) -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
docker-compose version # check 2.36.0 or above
sudo service docker start
cd <location of the code on the EC2 instance>
sudo docker-compose up -d # the -d is detacthed mode, i.e. makes sure it keeps running even when we close the connection
```
5. Edit the existing security group
- Navigate on the AWS console to EC2 > Security Groups and select the one associated with the EC2 instance
- Edit inbound rules and add a new inbound rule with type = Custom TCP, Protocol = TCP and Port range = 8500 (matches the frontend port), and source either your IP address if you only need to access it, otherwise 0.0.0.0/0 for the whole new world
- You should then be able to access it on the address http://<public ip address>:8500

#### Clean up / delete resources
- Terminate your EC2 instance
- See EBS > Volumes to check the volumes associated have been deleted
- See EC2 > Network interfaces to check they have been deleted
- See EC2 > Security Groups to delete any created

---
## Daily log
Note: Github Pilot/cursor/claude and other AI generating code was **not** used. Just good old internet searches.

**Day 1: Thursday 8th May 2025**
- Started by trying to read up on PyTorch - first time using it, and first time training up a machine learning model. Found a tutorial to follow along, and managed to get a PyTorch model trained and tested.
- Started to work on the Streamlit frontend. Also first time using it so trying it out, surprised how quickly you can get stuff up!

**Day 2: Friday 9th May 2025**
- Created a postgreSQL database and wrote code to store and view feedback records
- Finished the frontend interface, including drawable canvas and feedback input and display

**Day 3: Wednesday 14th May 2025**
- Split the codebase into different services (frontend, database and model), and ensure frontend all linked up through APIs
- Dockerised each of the different services and used docker-compose for the multi container setup

**Day 4: Thursday 15th May 2025**
- Set up AWS EC2 instance and deployed the codebase
- README write up

### Tasks
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
4. ✅ **Containerization with Docker**
- ✅ Use Docker to containerize:
    - ✅ The PyTorch model/service
    - ✅ The Streamlit web app
    - ✅ The PostgreSQL database
- ✅ Use Docker Compose to define your multi-container setup in a docker-compose.yml file.
5. ✅ **Deployment**
- ✅ Set up a self-managed server (e.g., Hetzner's basic instance) or any other environment where you can install Docker and control the deployment end-to-end.
- ✅ Deploy your containerized application to the server and make it accessible via a public IP or domain.
6. ✅ **Add project to GitHub**
- ✅ Add your project to GitHub.
- ✅ Make sure to include a README with a link to the live application.
- ✅ Share the link to your GitHub repository with us via the application form.

## Things I wish I had time to do
- Tests tests tests. Usually I code using TDD but since I was using tools I haven't used before (pytorch and streamlit), I deemed this to be Proof of Concept/a series of technical spikes so didn't add any tests.
- Obviously more reading up on machine learning algorithms and make the model more effective. And also use GPU instead of CPU as the device
- Deployment pipeline - rather than SSH'ing into the EC2 instance to deploy, I would want to have a proper deployment pipeline where it would automatically detect code changes and then deploy
- Set up HTTPS - add certificates etc needed. Link it up to a domain I own instead of the public IP address generated by AWS
- Use terraform to set up the AWS infrastructure
- Explore ways to get the Docker image size even smaller (especially the modelapp one, that is around 5GB currently. Managed to get it down from around 11GB but the smaller the better!) And related - see what the smallest AWS linux instance type can be without it erroring because of lack of CPU.
- Change the entire codebase to use the latest possible version of python. When I had a quick look, there were some package versions that weren't compatible with the latest version, but would be nice to explore further and upgrade
- A lot more love on the AWS front - e.g. setting up environment variables using secrets manager etc.
- Review how the frontend sends across the img as a buffer, then in the model service/API it converts it back into a numpy and then into a tensor. I'm not entirely convinced this is the most efficient way, and pretty sure I'm losing the quality of the input with all the conversions. Would like to revisit this.
- I'm sure lots more refactoring etc and other improvements, but the above is what is at the top of my head.

## Resources
- Learn Machine learning on pytorch: https://www.learnpytorch.io/03_pytorch_computer_vision/ 
- Multi stage docker builds with poetry and python: https://gabnotes.org/lighten-your-python-image-docker-multi-stage-builds/ 
- Deploying to an AWS EC2 instance (including installing Docker and docker compose): https://medium.com/@umairnadeem/deploy-to-aws-using-docker-compose-simple-210d71f43e67 