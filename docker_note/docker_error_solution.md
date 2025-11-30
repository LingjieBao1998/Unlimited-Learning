## ERROR: Got permission denied while trying to connect to the Docker daemon socket at unix:///var/run/docker.sock

The error occurs because Docker requires elevated permissions to access its Unix socket, located at /var/run/docker.sock. This socket communicates between the Docker client and the Docker daemon, which manages Docker containers and images. If your user **doesn’t have the right permissions**, Docker will refuse the connection.

1. Create the Docker Group
Docker typically requires `sudo` to execute commands, but by adding the current user to the Docker group, you can run Docker without needing sudo every time.
```bash
sudo groupadd docker
```
2. Add the User to the Docker Group
```bash
sudo usermod -aG docker ${USER}
```
This command adds your user to the `docker` group. Keep in mind that you’ll need to log out and log back in for the change to take effect.
3. Adjust Docker Socket Permissions
To ensure everything works correctly, adjust the permissions on the Docker socket itself. This step allows processes to communicate with Docker even if they’re not run by the `root` user or members of the `docker` group.
```bash
sudo chmod 666 /var/run/docker.sock
```
4. Restart the Docker Service
```bash
sudo systemctl restart docker
```
5. The Final Test
```bash
docker run -d -p 8000:8000 --name fastapi-app fastapi:latest
```

Docker is a powerful tool, but dealing with permission issues like this can be a real pain. Fortunately, the solution is simple: by adding your user to the `docker` group and adjusting permissions, you can avoid needing `sudo` for every Docker command.


> ref:https://medium.com/@wrefordmessi/how-to-fix-permission-denied-while-trying-to-connect-to-the-docker-daemon-socket-48933b9da2f4