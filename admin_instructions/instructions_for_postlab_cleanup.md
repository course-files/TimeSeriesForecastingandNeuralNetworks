# Cleanup Instructions

You can either:

- Wait until the end of the semester after the marks have been officially released to implement the cleanup in all the labs.
- Implement the cleanup once you are done with the lab **and you have pushed your commits to GitHub for grading**

The cleanup involves deleting the following from you development machine to save on space:

## Images to Delete

- servingmlmodels-flask-gunicorn-app
- servingmlmodels-nginx
- customized-ubuntu-server-smm

## Container Volumes to Delete

- From the Project Repository: `container-volumes\nginx\certs`
- From the Project Repository: `container-volumes\ubuntu\home-student`
- From Docker Volumes: `servingmlmodels_home-student`
- From Docker Volumes: `nginx-certs`
- From Docker Volumes: `nginx-frontend`

## Environment Variables in the `.env` file

- `.env`

## Python Virtual Environment

- `.venv`
