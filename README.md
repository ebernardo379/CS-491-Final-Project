# CS-491-Final-Project
Machine Learning Decision Tree Model Training and Testing Program, built with Python

Building and Testing Implemented with Github Actions, which runs the workflow on every push to main

Packaging and Deploying implemented with Docker and Docker Hub.
Packaging and Deploying workflow runs after the test workflow, and will build tine Dockerfile image and pushes the container to Docker Hub

To run code from Docker Hub:

docker build ernestob379/cs491-final-project-demo 

docker run ernestob379/cs491-final-project-demo
