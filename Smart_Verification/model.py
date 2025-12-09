












# # pip install huggingface_hub
# from huggingface_hub import hf_hub_download
# path = hf_hub_download(repo_id="Wespeaker/wespeaker-ecapa-tdnn512-LM",
#                        filename="voxceleb_ECAPA512_LM.onnx")
# print("Downloaded to:", path)













# import onnxruntime as ort

# sess = ort.InferenceSession("ecapa.onnx")
# inp = sess.get_inputs()[0]
# print("Model input name:", inp.name)
# print("Model input shape:", inp.shape)
