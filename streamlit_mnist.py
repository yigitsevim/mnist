{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "c11297c9-1f93-4f60-956e-209a7f731264",
   "metadata": {},
   "outputs": [
    {
     "ename": "AttributeError",
     "evalue": "Can't get attribute 'Net' on <module '__main__'>",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mAttributeError\u001b[0m                            Traceback (most recent call last)",
      "Cell \u001b[1;32mIn[14], line 55\u001b[0m\n\u001b[0;32m     51\u001b[0m         predict(model, categories, image)\n\u001b[0;32m     54\u001b[0m \u001b[38;5;28;01mif\u001b[39;00m \u001b[38;5;18m__name__\u001b[39m \u001b[38;5;241m==\u001b[39m \u001b[38;5;124m'\u001b[39m\u001b[38;5;124m__main__\u001b[39m\u001b[38;5;124m'\u001b[39m:\n\u001b[1;32m---> 55\u001b[0m     \u001b[43mmain\u001b[49m\u001b[43m(\u001b[49m\u001b[43m)\u001b[49m\n",
      "Cell \u001b[1;32mIn[14], line 45\u001b[0m, in \u001b[0;36mmain\u001b[1;34m()\u001b[0m\n\u001b[0;32m     43\u001b[0m \u001b[38;5;28;01mdef\u001b[39;00m \u001b[38;5;21mmain\u001b[39m():\n\u001b[0;32m     44\u001b[0m     st\u001b[38;5;241m.\u001b[39mtitle(\u001b[38;5;124m'\u001b[39m\u001b[38;5;124mPretrained model demo\u001b[39m\u001b[38;5;124m'\u001b[39m)\n\u001b[1;32m---> 45\u001b[0m     model \u001b[38;5;241m=\u001b[39m \u001b[43mload_model\u001b[49m\u001b[43m(\u001b[49m\u001b[43m)\u001b[49m\n\u001b[0;32m     46\u001b[0m     categories \u001b[38;5;241m=\u001b[39m [\u001b[38;5;241m0\u001b[39m,\u001b[38;5;241m1\u001b[39m,\u001b[38;5;241m2\u001b[39m,\u001b[38;5;241m3\u001b[39m,\u001b[38;5;241m4\u001b[39m,\u001b[38;5;241m5\u001b[39m,\u001b[38;5;241m6\u001b[39m,\u001b[38;5;241m7\u001b[39m,\u001b[38;5;241m8\u001b[39m,\u001b[38;5;241m9\u001b[39m]\n\u001b[0;32m     47\u001b[0m     image \u001b[38;5;241m=\u001b[39m load_image()\n",
      "Cell \u001b[1;32mIn[14], line 21\u001b[0m, in \u001b[0;36mload_model\u001b[1;34m()\u001b[0m\n\u001b[0;32m     20\u001b[0m \u001b[38;5;28;01mdef\u001b[39;00m \u001b[38;5;21mload_model\u001b[39m():\n\u001b[1;32m---> 21\u001b[0m     model \u001b[38;5;241m=\u001b[39m \u001b[43mtorch\u001b[49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43mload\u001b[49m\u001b[43m(\u001b[49m\u001b[38;5;124;43m'\u001b[39;49m\u001b[38;5;124;43mmnist_model.pt\u001b[39;49m\u001b[38;5;124;43m'\u001b[39;49m\u001b[43m)\u001b[49m\n\u001b[0;32m     22\u001b[0m     model\u001b[38;5;241m.\u001b[39meval()\n\u001b[0;32m     23\u001b[0m     \u001b[38;5;28;01mreturn\u001b[39;00m model\n",
      "File \u001b[1;32m~\\anaconda3\\envs\\deeplearning\\lib\\site-packages\\torch\\serialization.py:789\u001b[0m, in \u001b[0;36mload\u001b[1;34m(f, map_location, pickle_module, weights_only, **pickle_load_args)\u001b[0m\n\u001b[0;32m    787\u001b[0m             \u001b[38;5;28;01mexcept\u001b[39;00m \u001b[38;5;167;01mRuntimeError\u001b[39;00m \u001b[38;5;28;01mas\u001b[39;00m e:\n\u001b[0;32m    788\u001b[0m                 \u001b[38;5;28;01mraise\u001b[39;00m pickle\u001b[38;5;241m.\u001b[39mUnpicklingError(UNSAFE_MESSAGE \u001b[38;5;241m+\u001b[39m \u001b[38;5;28mstr\u001b[39m(e)) \u001b[38;5;28;01mfrom\u001b[39;00m \u001b[38;5;28mNone\u001b[39m\n\u001b[1;32m--> 789\u001b[0m         \u001b[38;5;28;01mreturn\u001b[39;00m _load(opened_zipfile, map_location, pickle_module, \u001b[38;5;241m*\u001b[39m\u001b[38;5;241m*\u001b[39mpickle_load_args)\n\u001b[0;32m    790\u001b[0m \u001b[38;5;28;01mif\u001b[39;00m weights_only:\n\u001b[0;32m    791\u001b[0m     \u001b[38;5;28;01mtry\u001b[39;00m:\n",
      "File \u001b[1;32m~\\anaconda3\\envs\\deeplearning\\lib\\site-packages\\torch\\serialization.py:1131\u001b[0m, in \u001b[0;36m_load\u001b[1;34m(zip_file, map_location, pickle_module, pickle_file, **pickle_load_args)\u001b[0m\n\u001b[0;32m   1129\u001b[0m unpickler \u001b[38;5;241m=\u001b[39m UnpicklerWrapper(data_file, \u001b[38;5;241m*\u001b[39m\u001b[38;5;241m*\u001b[39mpickle_load_args)\n\u001b[0;32m   1130\u001b[0m unpickler\u001b[38;5;241m.\u001b[39mpersistent_load \u001b[38;5;241m=\u001b[39m persistent_load\n\u001b[1;32m-> 1131\u001b[0m result \u001b[38;5;241m=\u001b[39m \u001b[43munpickler\u001b[49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43mload\u001b[49m\u001b[43m(\u001b[49m\u001b[43m)\u001b[49m\n\u001b[0;32m   1133\u001b[0m torch\u001b[38;5;241m.\u001b[39m_utils\u001b[38;5;241m.\u001b[39m_validate_loaded_sparse_tensors()\n\u001b[0;32m   1135\u001b[0m \u001b[38;5;28;01mreturn\u001b[39;00m result\n",
      "File \u001b[1;32m~\\anaconda3\\envs\\deeplearning\\lib\\site-packages\\torch\\serialization.py:1124\u001b[0m, in \u001b[0;36m_load.<locals>.UnpicklerWrapper.find_class\u001b[1;34m(self, mod_name, name)\u001b[0m\n\u001b[0;32m   1122\u001b[0m         \u001b[38;5;28;01mpass\u001b[39;00m\n\u001b[0;32m   1123\u001b[0m mod_name \u001b[38;5;241m=\u001b[39m load_module_mapping\u001b[38;5;241m.\u001b[39mget(mod_name, mod_name)\n\u001b[1;32m-> 1124\u001b[0m \u001b[38;5;28;01mreturn\u001b[39;00m \u001b[38;5;28;43msuper\u001b[39;49m\u001b[43m(\u001b[49m\u001b[43m)\u001b[49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43mfind_class\u001b[49m\u001b[43m(\u001b[49m\u001b[43mmod_name\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[43mname\u001b[49m\u001b[43m)\u001b[49m\n",
      "\u001b[1;31mAttributeError\u001b[0m: Can't get attribute 'Net' on <module '__main__'>"
     ]
    }
   ],
   "source": [
    "import io\n",
    "import os\n",
    "from PIL import Image\n",
    "import streamlit as st\n",
    "import torch\n",
    "from torchvision import transforms\n",
    "import wget\n",
    "\n",
    "\n",
    "def load_image():\n",
    "    uploaded_file = st.file_uploader(label='Pick an image to test')\n",
    "    if uploaded_file is not None:\n",
    "        image_data = uploaded_file.getvalue()\n",
    "        st.image(image_data)\n",
    "        return Image.open(io.BytesIO(image_data))\n",
    "    else:\n",
    "        return None\n",
    "\n",
    "\n",
    "def load_model():\n",
    "    model = torch.load('mnist_model.pt')\n",
    "    model.eval()\n",
    "    return model\n",
    "\n",
    "\n",
    "def predict(model, categories, image):\n",
    "    preprocess = transforms.Compose([\n",
    "        transforms.ToTensor(),\n",
    "    ])\n",
    "    input_tensor = preprocess(image)\n",
    "    input_batch = input_tensor.unsqueeze(0)\n",
    "\n",
    "    with torch.no_grad():\n",
    "        output = model(input_batch)\n",
    "\n",
    "    probabilities = torch.nn.functional.softmax(output[0], dim=0)\n",
    "\n",
    "    top5_prob, top5_catid = torch.topk(probabilities, 5)\n",
    "    for i in range(top5_prob.size(0)):\n",
    "        st.write(categories[top5_catid[i]], top5_prob[i].item())\n",
    "\n",
    "\n",
    "def main():\n",
    "    st.title('Pretrained model demo')\n",
    "    model = load_model()\n",
    "    categories = [0,1,2,3,4,5,6,7,8,9]\n",
    "    image = load_image()\n",
    "    result = st.button('Run on image')\n",
    "    if result:\n",
    "        st.write('Calculating results...')\n",
    "        predict(model, categories, image)\n",
    "\n",
    "\n",
    "if __name__ == '__main__':\n",
    "    main()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "1e8f582e-e42a-4b67-a147-d254cb19ec32",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.16"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
