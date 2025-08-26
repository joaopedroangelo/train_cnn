import os
from PIL import Image
from torchvision import transforms

# ========================
# Configurações
# ========================
INPUT_DIR = './dataset/temp'
OUTPUT_DIR = './dataset/aumentado'
IMG_SIZE = (112, 112)
NUM_AUGMENTED = 100  # imagens geradas por imagem original

# ========================
# Transformações para Aumento
# ========================
augment = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(25),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
])

# ========================
# Função para Aumentar Dados
# ========================
def augment_images(input_dir, output_dir, num_augmented, transform):
    os.makedirs(output_dir, exist_ok=True)

    for img_name in os.listdir(input_dir):
        img_path = os.path.join(input_dir, img_name)
        
        # Ignora arquivos que não são imagens
        if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        image = Image.open(img_path).convert('RGB')
        base_name, _ = os.path.splitext(img_name)

        for i in range(num_augmented):
            aug_img = transform(image)
            aug_img.save(os.path.join(output_dir, f"{base_name}_aug_{i}.jpg"))

    print("Aumento de dados concluído!")

# ========================
# Execução Principal
# ========================
if __name__ == "__main__":
    augment_images(INPUT_DIR, OUTPUT_DIR, NUM_AUGMENTED, augment)
