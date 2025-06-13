# Django + Docker + PostgreSQL

## Description

This project runs a Django web application with PostgreSQL using Docker and Docker Compose.

## Stack

- Python 3.11 with Django
- PostgreSQL (host port: `5431`, container port: `5432`)
- Automatic migrations on container start

## Getting Started

1. Build and start the containers:

```bash
docker-compose up --build
```
