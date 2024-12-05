```bash

docker run \
    --name neo4j_apoc \
    --publish=7474:7474 --publish=7687:7687 \
    --volume=$HOME/neo4j/data:/data \
    --env NEO4J_AUTH=neo4j/your_password \
    --env NEO4JLABS_PLUGINS='["apoc"]' \
    neo4j
```

To install neo4j, 