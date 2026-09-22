from langchain_text_splitters import RecursiveCharacterTextSplitter

text ="""
Exploring the VAE
We’ll now take a look at the output from the VAE and each submodel and then see
how the VAE can be used to generate completely new track observations.
The VAE model
If we feed the VAE with an observation, it is able to accurately reconstruct the origi‐
nal image, as shown in Figure 12-7. This is useful to visually check that the VAE is
working correctly.
Figure 12-7. The input and output from the VAE model
The encoder models
If we feed the encode_mu_logvar model with an observa."""

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 100,
    chunk_overlap = 20
)
chunks = splitter.split_text(text)

for i,chunk in enumerate(chunks):
    print(f"\n...chunk {i+1} ---")
    print(chunk)