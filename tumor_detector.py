import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image

import torch
# from torchvision.utils import draw_bounding_boxes
# from torchvision.transforms.functional import pil_to_tensor

from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator

def draw(image, pred) -> None: 
    # st.write(pred[0].boxes.conf.item())
    # st.write(pred[0].boxes)
    annotator = Annotator(image)
    boxes = pred[0].boxes
    for box in boxes:
        b = box.xyxy[0]
        c = box.cls
        annotator.box_label(b, f"{model.names[int(c)]} {round(box.conf.item(), 3)}" )

    res = annotator.result()
    st.image(res, "Detections with YOLOv8", use_column_width=True)

    # st.write(pred.xyxyn[0][:, :-2])
    # img_with_boxes = draw_bounding_boxes(
    #     pil_to_tensor(image),
    #     boxes = pred.xyxyn[0][:, :-2] * image.width, 
    #     labels = pred.pandas().xyxyn[0]["name"],
    #     colors = "red",
    #     width = 2,
    #     font_size = 30
    # ).detach().numpy().transpose(1, 2, 0)
    
    # fig, ax = plt.subplots(figsize=(12, 12))
    # plt.imshow(img_with_boxes)
    # plt.xticks([], [])
    # plt.yticks([], [])

    # plt.setp(ax.spines.values(), visible=False) 

    # st.pyplot(fig, use_container_width=True)


st.set_page_config(page_title="Brain tumor detector")
st.title("Brain tumor detector")

page_bg_img = '''
<style>
.stAppViewContainer {
    background-image: url("https://formaspace.com/wp-content/uploads/2023/10/lab-artificial-ai.jpeg");
    background-size: cover;
}
.stAppViewBlockContainer {
    -webkit-box-align: center;
    margin-top: 100px;
    margin-bottom: 50px;
    padding-top: 10px;
    background-color: rgb(240, 242, 246);
    border-radius: 0.5rem;
    color: rgb(49, 51, 63);
}
</style>
'''

st.markdown(page_bg_img, unsafe_allow_html=True)

try:
    # model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, trust_repo=True)
    model = YOLO("best.pt", verbose=False)
    model.eval(verbose=False)
    print('Model successfully loaded')
except Exception as e:
    print(e)

uploaded_image = st.file_uploader(" ", type=['png', 'jpg', 'jpeg'])

if uploaded_image:
    image = Image.open(uploaded_image)
    st.image(image, caption="Here's what you uploaded", use_column_width=True)

    if st.button("Detect"):
        prediction = model.predict(image, verbose=False)
        # st.write(prediction.pandas().xyxyn[0])
        draw(image, prediction)
