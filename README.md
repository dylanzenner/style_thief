# Steps for replication (Local Host)


### Step 1: 
- clone the repo to your local machine

### Step 2:
- open up the file in your editor of choice.
- open a terminal session
- run the following command

```
docker-compose up
```

### Step 3:
- Stylize your favorite images!

Note: The video below is sped up. Currently the algorithm taken ~5 - 7 minutes to run the first time you use it. Subsequent runs take ~ 3 - 4 minutes

https://user-images.githubusercontent.com/54154602/184963072-52de6b5a-80a3-4ffc-a570-e453352d2b6d.mov


# Steps for replication (AWS ECS)

### Step 1:
- clone the repo to your local machine

### Step 2:
- Build and push the frontend and backend containers to ECR

### Step 3:
- Within your code editor create a new ecs context with
```
docker context create ecs style-thief-context
```
- follow the terminal instructions to give permissions
- run the following command
```
docker context use style-thief-context
```
- run the following command
```
docker compose up
```
- run the following command
```
docker ps
```
This is show you where Style Thief is running
- add ```:3000``` to the end of the link where Style Thief is running and navigate to the link!
- Note: You'll also have to change the fetch request URL in the Frontend to point to this endpoint.
