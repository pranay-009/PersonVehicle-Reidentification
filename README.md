# PersonVehicle-Reidentification
The repository contains the work regarding Person and Vehicle Reidentification using the Segmentation and Symmetry concept

## Installation Guide

**To set up and run this project locally, follow the steps below**:
- Install Python Version ```3.10.5```

- **Create a Virtual Environment**
  ```bash
    #create a virtual environment (name it anything you like ) I am naming it venv
    python -m venv venv
    #linux or mac
    source venv/bin/activate
    #windows
    .\venv\Scripts\activate
  ```
- **Clone Repository**
  ```
  #clone
  https://github.com/pranay-009/PersonVehicle-Reidentification.git
  #move to repo directory
  cd PersonVehicle-Reidentification
  ```
- **Installation through pip**
  ```
  # move to PersonVehicle-Reidentification directory
  pip install -r requirements.txt
  ```
- **Using  Docker(Recomended)**
  ```
  # move to the working directory
  cd PersonVehicle-Reidentification
  docker build -t <image_name>:<version>  .
  #replace image_name with version number 
  docker run -p 8080:8080 --name <container_name> <image_name>
  #replace the container_name with the container name yoou want to use and the image created in Docker
  ```
##
  



