"""Olist veri setini kagglehub ile indirir ve klasör yolunu yazdırır (hesap gerekmez).

    python scripts/download_data.py
"""
import kagglehub

if __name__ == "__main__":
    print(kagglehub.dataset_download("olistbr/brazilian-ecommerce"))
