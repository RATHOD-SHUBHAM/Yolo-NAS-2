# Todo:  Load Image

import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import streamlit as st
import os

# hide hamburger and customize footer
hide_menu= """
<style>

#MainMenu {
    visibility:hidden;
}

footer{
    visibility:visible;
}

footer:after{
    content: 'With 🫶️ from Shubham Shankar.';
    display:block;
    position:relative;
    color:grey;
    padding;5px;
    top:3px;
}
</style>

"""
# Styling ----------------------------------------------------------------------

st.image("/Users/shubhamrathod/PycharmProjects/SAM_streamlit/icon.jpg", width=85)
st.title("NASAM")
st.subheader("Object Detection and Mask")
st.markdown(hide_menu, unsafe_allow_html=True)

# Intro ----------------------------------------------------------------------

st.write(
    """

    Hi 👋, I'm **:red[Shubham Shankar]**, and welcome to my **:green[Object Detection + MASK Application]**! :rocket: This program makes use of **:blue[YOLO-NAS]** and **:orange[SAM]** model, 
    which was specially trained using the **:violet[Roboflow]** dataset.  ✨

    """
)

st.markdown('---')

st.write(
    """
    ### App Interface!!

    :dog: The web app has an easy-to-use interface. 

    1] **:green[Upload File]**: Upload an image using the provided button. The app will perform inference on the image using a  machine learning model and display the results.

    2] **:violet[Confidence Threshold]**: Adjust the confidence threshold to get a better result.

    """
)

st.markdown('---')

st.error(
    """
    Connect with me on [**Github**](https://github.com/RATHOD-SHUBHAM) and [**LinkedIn**](https://www.linkedin.com/in/shubhamshankar/). ✨
    """,
    icon="🧟‍♂️",
)

st.markdown('---')

# remove file in a folder ----------------------------------------------------------------------
import shutil
folder = '/Users/shubhamrathod/PycharmProjects/SAM_streamlit/op_detection'
for filename in os.listdir(folder):
    file_path = os.path.join(folder, filename)
    try:
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)
    except Exception as e:
        print('Failed to delete %s. Reason: %s' % (file_path, e))

# Save File ------------------------------------------------------------------------------------------
def save_uploadedfile(uploadedfile):
    with open(os.path.join("/Users/shubhamrathod/PycharmProjects/SAM_streamlit/ip_image", uploadedfile.name), "wb") as f:
        f.write(uploadedfile.getbuffer())

# Model ----------------------------------------------------------------------------------------------------
image_file = st.file_uploader("Upload An Image", type=['png','jpeg','jpg'])
if image_file is not None:
    file_details = {"FileName":image_file.name,"FileType":image_file.type}

    conf_threshold = st.slider('Confidence Threshold', min_value=0.0, max_value=1.0, value=0.35)

    if st.button('RUN'):

        st.write(file_details)
        save_uploadedfile(image_file)



        col1, col2, col3 = st.columns(3)

        image = cv2.imread("/Users/shubhamrathod/PycharmProjects/SAM_streamlit/ip_image/" + image_file.name)

        with col1:
            st.text("Raw Image")
            st.image(image)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Todo: Development

        from super_gradients.training import models

        # define class name
        class_names = ['Car boot', 'Car hood', 'Driver-s door - -F-R-', 'Fender - -F-L-', 'Fender - -F-R-', 'Fender - -R-L-', 'Fender - -R-R-', 'Front bumper', 'Headlight - -L-', 'Headlight - -R-', 'Passenger-s door - -F-L-', 'Passenger-s door - -R-L-', 'Passenger-s door - -R-R-', 'Rear bumper', 'Rear light - -L-', 'Rear light - -R-', 'Side bumper - -L-', 'Side bumper - -R-', 'Side mirror - -L-', 'Side mirror - -R-']

        # Todo: Get the model

        device = 'cuda' if torch.cuda.is_available() else "cpu"
        model_nas = models.get('yolo_nas_l',
                               num_classes= 20,
                               checkpoint_path='/Users/shubhamrathod/PycharmProjects/SAM_streamlit/nas_weight/ckpt_best.pth')



        # Todo: Object detection prediction
        model_nas.predict(image, conf = conf_threshold).save('/Users/shubhamrathod/PycharmProjects/SAM_streamlit/op_detection')

        with col2:
            st.text("Detection Output")
            st.image('/Users/shubhamrathod/PycharmProjects/SAM_streamlit/op_detection/pred_0.jpg')

        # Todo: Get BBOX
        model_pred = list(model_nas.predict(image, conf = conf_threshold)._images_prediction_lst)

        bboxes_xyxy = model_pred[0].prediction.bboxes_xyxy.tolist()

        # Todo: SAM

        def show_mask(mask, ax, random_color=False):
            if random_color:
                color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
            else:
                color = np.array([30/255, 144/255, 255/255, 0.6])
            h, w = mask.shape[-2:]
            mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
            ax.imshow(mask_image)

        def show_points(coords, labels, ax, marker_size=375):
            pos_points = coords[labels==1]
            neg_points = coords[labels==0]
            ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
            ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

        def show_box(box, ax):
            x0, y0 = box[0], box[1]
            w, h = box[2] - box[0], box[3] - box[1]
            ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))

        from segment_anything import sam_model_registry, SamPredictor
        from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

        sam_checkpoint = "/Users/shubhamrathod/PycharmProjects/SAM_streamlit/sam_weight/sam_vit_h_4b8939.pth"
        model_type = "vit_h"

        device = 'cuda' if torch.cuda.is_available() else "cpu"

        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device=device)

        predictor = SamPredictor(sam)
        mask_generator = SamAutomaticMaskGenerator(sam)

        # Todo: SAM predictor
        predictor.set_image(image)

        tensor_box = torch.tensor(bboxes_xyxy, device=predictor.device)

        transformed_boxes = predictor.transform.apply_boxes_torch(tensor_box, image.shape[:2])

        batch_masks, batch_scores, batch_logits = predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False,
        )

        plt.figure(figsize=(10, 10))
        plt.imshow(image)

        for mask in batch_masks:
            show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)

        plt.axis('off')
        plt.savefig('my_image.jpg')

        with col3:
            st.text("Masked Output")
            st.image('my_image.jpg')