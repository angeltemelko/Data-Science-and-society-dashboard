# Store Sales Forecasting

A working dashboard that would help waltmart professionals make key decisions on the current and future trends of holiday sales. 

## Installation

You first need to install Docker desktop to your PC from the following link: 

- Windows:[Docker Desktop](https://www.docker.com/products/docker-desktop)

- Linux: install both Engine and Compose
     1. [Docker Engine](https://docs.docker.com/engine/install/)
     2. [Docker Compose](https://docs.docker.com/compose/install/#install-compose-on-linux-systems)

- Open Docker Desktop(Linux: make sure the daemon process is running)
- Download or clone the project from [GitHub](https://github.com/angeltemelko/bi_dashboard) 
## Usage

First, you need to open a Termina(CMD) where the Dockerfile is located.

-Optional-

NOTE: Sometimes running just docker-compose up will not insall all the packages.Therefore try this optional command

To force install requirements, run in Terminal: 
```bash
docker-compose build --no-cache
```
This takes around a minute, but can fix issues with modules not being installed in the docker instance

To start the dashboard, run in Terminal:
```bash
docker-compose up
```
To close the dashboard, run in Terminal:

```bash
docker-compose down
```

Make sure that no other application is running on port 8080

To see the dashboard in your browser, write: http://localhost:8080.

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License
[MIT](https://choosealicense.com/licenses/mit/)
