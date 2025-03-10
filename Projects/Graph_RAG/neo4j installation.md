# Running Neo4j in Docker with Custom Port Mapping

The following command runs a Neo4j container using Docker, with custom port mappings to avoid conflicts with other processes or containers.

## Command:
```bash
docker run \
  --name neo4j \
  -p 7475:7474 -p 7688:7687 \
  -e NEO4J_AUTH=neo4j/myPassword \
  -e NEO4JLABS_PLUGINS='["apoc"]' \
  -v $HOME/neo4j/data:/data \
  -v $HOME/neo4j/logs:/logs \
  -v $HOME/neo4j/import:/import \
  -v $HOME/neo4j/plugins:/plugins \
  neo4j:latest

```

## Explanation of Parameters:

### Container Name
- `--name neo4j`:
  Specifies the name of the container for easier management.

### Port Mapping
- `-p 7475:7474`:
  Maps port `7474` inside the container (Neo4j HTTP interface) to port `7475` on the host.
- `-p 7688:7687`:
  Maps port `7687` inside the container (Bolt protocol) to port `7688` on the host.

### Authentication
- `-e NEO4J_AUTH=neo4j/myPassword`:
  Sets the authentication credentials for the Neo4j instance. Replace `myPassword` with your desired password.

### Volume Mapping
- `-v $HOME/neo4j/data:/data`:
  Maps the host directory `$HOME/neo4j/data` to the container's `/data` directory for persistent database storage.
- `-v $HOME/neo4j/logs:/logs`:
  Maps the host directory `$HOME/neo4j/logs` to the container's `/logs` directory for log storage.
- `-v $HOME/neo4j/import:/import`:
  Maps the host directory `$HOME/neo4j/import` to the container's `/import` directory for importing data.
- `-v $HOME/neo4j/plugins:/plugins`:
  Maps the host directory `$HOME/neo4j/plugins` to the container's `/plugins` directory for managing plugins.

### Image
- `neo4j:latest`:
  Specifies the Neo4j Docker image to use. `latest` pulls the most recent version of the image.

## Notes:
- Ensure that the directories specified in the `-v` options exist on the host.
- This setup avoids conflicts by using ports `7475` and `7688` instead of the default ports `7474` and `7687`.
