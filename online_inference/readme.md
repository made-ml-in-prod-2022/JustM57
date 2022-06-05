docker pull justm57/online_inference:v1 .
docker run -p 5757:5757 justm57/online_inference:v1
python make_request.py
