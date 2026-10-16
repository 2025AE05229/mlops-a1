# Kubernetes deployment (kind + podman)

Run all commands from the `Assignment/` directory. The kind cluster here is `heart-disease-mlops`

```bash
# 1. Build the image
podman build -t heart-disease-api:local .

# 2. Load it into kind.
podman save -o /tmp/heart-disease-api.tar localhost/heart-disease-api:local
kind load image-archive /tmp/heart-disease-api.tar --name heart-disease-mlops

# 3. Apply the manifests
kubectl apply -k deployment/
kubectl -n heart-disease rollout status deployment/heart-disease-api

# 4. Reach the service
kubectl -n heart-disease port-forward svc/heart-disease-api 8080:80
```

Or via the Contour route in [`httpproxy.yaml`](httpproxy.yaml) (no port-forward needed; kind maps node port 80 to host port 9090):

```bash
curl http://heart-disease.localtest.me:9090/health
```

Smoke test (in a new terminal):

```bash
curl http://localhost:8080/health
curl -u <your-username>:<your-password> -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"age":63,"sex":1,"cp":1,"trestbps":145,"chol":233,"fbs":1,"restecg":2,"thalach":150,"exang":0,"oldpeak":2.3,"slope":3,"ca":0,"thal":6}'
```

After rebuilding the image, repeat steps 1-2 and run
`kubectl -n heart-disease rollout restart deployment/heart-disease-api`.

Teardown: `kubectl delete -k deployment/`
