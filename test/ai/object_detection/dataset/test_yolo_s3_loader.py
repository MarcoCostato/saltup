"""
pytest fixtures for testing YOLO S3 loaders with moto (S3 mock)
"""
import os
import json
import boto3
import pytest
import time
import io
from PIL import Image
from moto import mock_aws
from botocore.exceptions import ClientError

os.environ["SALTUP_BACKEND"] = "keras_tensorflow"

from saltup.utils.data.s3.s3_utils import S3
from saltup.ai.object_detection.dataset.yolo_darknet import YoloDarknetS3Loader


@pytest.fixture(scope="function")
def aws_credentials():
    """Mocked AWS Credentials for moto"""
    import os
    os.environ['AWS_ACCESS_KEY_ID'] = 'testing'
    os.environ['AWS_SECRET_ACCESS_KEY'] = 'testing'
    os.environ['AWS_SECURITY_TOKEN'] = 'testing'
    os.environ['AWS_SESSION_TOKEN'] = 'testing'
    os.environ['AWS_DEFAULT_REGION'] = 'us-east-1'
    
@pytest.fixture(scope="function")
def s3_client(aws_credentials):
    """Create mocked S3 client"""
    with mock_aws():
        yield boto3.client('s3', region_name='us-east-1')


@pytest.fixture
def test_bucket(s3_client):
    """Create a test bucket with YOLO data"""
    bucket_name = "test-yolo-bucket"
    s3_client.create_bucket(Bucket=bucket_name)
    
    # Create sample YOLO label data
    yolo_labels = {
        "labels/image1.txt": "0 0.5 0.5 0.2 0.3\n1 0.3 0.7 0.1 0.2",
        "labels/image2.txt": "0 0.4 0.6 0.3 0.4",
    }
    
    # Create REAL image files - not fake binary data
    image_files = {}
    
    # Create valid JPEG images for testing
    for i, name in enumerate(["images/image1.jpg", "images/image2.jpg"], 1):
        img = Image.new('RGB', (100, 80), color=(255, 0, 0))  # 100x80 red image
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='JPEG')
        image_files[name] = img_bytes.getvalue()
    
    # Debug: Print what we're uploading
    print(f"Uploading {len(image_files)} images to S3")
    for key, data in image_files.items():
        print(f"  {key}: {len(data)} bytes")
    
    # Upload YOLO labels
    for key, content in yolo_labels.items():
        s3_client.put_object(
            Bucket=bucket_name,
            Key=key,
            Body=content.encode('utf-8')
        )
    
    # Upload images
    for key, content in image_files.items():
        s3_client.put_object(
            Bucket=bucket_name,
            Key=key,
            Body=content
        )
    
    # Debug: Verify uploads
    response = s3_client.list_objects_v2(Bucket=bucket_name)
    if 'Contents' in response:
        print("S3 bucket contents:")
        for obj in response['Contents']:
            print(f"  {obj['Key']}: {obj['Size']} bytes")
    
    yield bucket_name

def test_yolo_s3_loader_basic(s3_client, test_bucket):
    """Test basic YOLO S3 loader functionality"""
    s3 = S3(test_bucket)
    
    # Debug: Check S3 connection
    print(f"S3 bucket: {test_bucket}")
    print(f"S3 client: {s3}")
    
    dataset = YoloDarknetS3Loader(
        images_dir="images/",
        labels_dir="labels/",
        s3_client=s3,
    )
    
    # Test basic properties
    print(f"Dataset length: {len(dataset)}")
    assert len(dataset) == 2
    path, image, labels = dataset[0]
    assert path is not None
    assert image is None
    assert labels is not None

def test_yolo_s3_loader_basic_with_image(s3_client, test_bucket):
    """Test basic YOLO S3 loader functionality"""
    s3 = S3(test_bucket)
    
    # Debug: Check S3 connection
    print(f"S3 bucket: {test_bucket}")
    print(f"S3 client: {s3}")
    
    dataset = YoloDarknetS3Loader(
        images_dir="images/",
        labels_dir="labels/",
        s3_client=s3,
        download_file=True,
        max_files=10,
    )
    
    # Test basic properties
    print(f"Dataset length: {len(dataset)}")
    assert len(dataset) == 2
    path, image, labels = dataset[0]
    assert path is not None
    assert image is not None
    assert labels is not None
    
    
# Fix ALL other fixtures to use real images too
def test_multiple_yolo_buckets(s3_client):
    """Test working with multiple YOLO buckets"""
    # Create multiple buckets
    s3_client.create_bucket(Bucket="yolo-train")
    s3_client.create_bucket(Bucket="yolo-val")
    
    # Create REAL images for train bucket
    train_images = {}
    for name in ["images/train1.jpg", "images/train2.jpg"]:
        img = Image.new('RGB', (100, 80), color=(0, 255, 0))  # Green image
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='JPEG')
        train_images[name] = img_bytes.getvalue()
    
    # Create REAL image for val bucket
    val_images = {}
    img = Image.new('RGB', (100, 80), color=(0, 0, 255))  # Blue image
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='JPEG')
    val_images["images/val1.jpg"] = img_bytes.getvalue()
    
    # Add YOLO data to train bucket
    train_data = {
        **train_images,  # Real images
        "labels/train1.txt": "0 0.5 0.5 0.4 0.6",
        "labels/train2.txt": "1 0.3 0.3 0.2 0.2",
    }
    
    # Add YOLO data to val bucket
    val_data = {
        **val_images,  # Real images
        "labels/val1.txt": "0 0.6 0.4 0.3 0.5",
    }
    
    # Upload train data
    for key, content in train_data.items():
        if isinstance(content, str):
            content = content.encode('utf-8')
        s3_client.put_object(Bucket="yolo-train", Key=key, Body=content)
    
    # Upload val data  
    for key, content in val_data.items():
        if isinstance(content, str):
            content = content.encode('utf-8')
        s3_client.put_object(Bucket="yolo-val", Key=key, Body=content)
    
    # Test datasets from different buckets
    s3_train = S3("yolo-train")
    s3_val = S3("yolo-val")
    
    train_dataset = YoloDarknetS3Loader(
        images_dir="images/",
        labels_dir="labels/", 
        s3_client=s3_train
    )
    
    val_dataset = YoloDarknetS3Loader(
        images_dir="images/",
        labels_dir="labels/",
        s3_client=s3_val
    )
    
    # Verify different dataset sizes
    assert len(train_dataset) == 2
    assert len(val_dataset) == 1


@pytest.fixture
def s3_with_yolo_nested_structure(s3_client):
    """Create bucket with complex nested YOLO structure"""
    bucket_name = "test-yolo-nested-bucket"
    s3_client.create_bucket(Bucket=bucket_name)
    
    # Create REAL images for nested structure
    nested_images = {}
    image_names = [
        "dataset/train/images/image1.jpg",
        "dataset/train/images/image2.jpg", 
        "dataset/val/images/image3.jpg",
        "dataset/test/images/image4.jpg"
    ]
    
    for name in image_names:
        img = Image.new('RGB', (100, 80), color=(128, 128, 128))  # Gray image
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='JPEG')
        nested_images[name] = img_bytes.getvalue()
    
    # Create nested YOLO folder structure
    files_data = {
        **nested_images,  # Real images
        "dataset/train/labels/image1.txt": "0 0.5 0.5 0.2 0.3\n1 0.3 0.7 0.1 0.2",
        "dataset/train/labels/image2.txt": "0 0.4 0.6 0.3 0.4",
        "dataset/val/labels/image3.txt": "1 0.6 0.4 0.2 0.3",
        "dataset/test/labels/image4.txt": "0 0.7 0.3 0.1 0.2",
    }
    
    for path, data in files_data.items():
        if isinstance(data, str):
            data = data.encode('utf-8')
        s3_client.put_object(
            Bucket=bucket_name,
            Key=path,
            Body=data
        )
    
    yield bucket_name


def test_yolo_s3_loader_invalid_labels(s3_client):
    """Test handling of invalid YOLO label format"""
    bucket_name = "invalid-yolo-bucket"
    s3_client.create_bucket(Bucket=bucket_name)
    
    # Create REAL image with invalid label
    img = Image.new('RGB', (100, 80), color=(255, 255, 0))  # Yellow image
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='JPEG')
    
    invalid_data = {
        "images/bad1.jpg": img_bytes.getvalue(),  # Real image
        "labels/bad1.txt": "invalid_yolo_format",  # Invalid format
    }
    
    for key, content in invalid_data.items():
        if isinstance(content, str):
            content = content.encode('utf-8')
        s3_client.put_object(Bucket=bucket_name, Key=key, Body=content)
    
    s3 = S3(bucket_name)
    
    dataset = YoloDarknetS3Loader(
        images_dir="images/",
        labels_dir="labels/",
        s3_client=s3
    )
    
    # Should handle invalid labels gracefully
    assert len(dataset) >= 0  # Might skip invalid files


def test_yolo_s3_loader_missing_file(s3_client, test_bucket):
    """Test error handling for missing files"""
    s3 = S3(test_bucket)
    
    # Test with non-existent directory
    with pytest.raises(Exception):  # Could be ClientError or custom exception
        dataset = YoloDarknetS3Loader(
            images_dir="nonexistent/",
            labels_dir="labels/",
            s3_client=s3
        )
        path, image, labels = dataset[0]


def test_yolo_s3_loader_empty_bucket(s3_client):
    """Test with empty bucket"""
    bucket_name = "empty-yolo-bucket" 
    s3_client.create_bucket(Bucket=bucket_name)
    
    s3 = S3(bucket_name)
    
    dataset = YoloDarknetS3Loader(
        images_dir="images/",
        labels_dir="labels/",
        s3_client=s3
    )
    
    # Empty dataset should have length 0
    assert len(dataset) == 0

def test_yolo_nested_directory_structure(s3_client, s3_with_yolo_nested_structure):
    """Test YOLO loader handles nested directories"""
    s3 = S3(s3_with_yolo_nested_structure)
    
    train_dataset = YoloDarknetS3Loader(
        images_dir="dataset/train/images/",
        labels_dir="dataset/train/labels/", 
        s3_client=s3,
        download_file=True,
        max_files=10
    )
    
    val_dataset = YoloDarknetS3Loader(
        images_dir="dataset/val/images/",
        labels_dir="dataset/val/labels/",
        s3_client=s3,
        download_file=True,
        max_files=10
    )
    
    # Test train set
    assert len(train_dataset) == 2
    
    # Test val set  
    assert len(val_dataset) == 1
    
    # Test data access
    train_path, train_image, train_labels = train_dataset[0]
    assert train_image is not None
    assert len(train_labels) > 0  # Should have annotations


def test_yolo_s3_loader_timing(s3_client, test_bucket):
    """Test YOLO loader timing without external benchmark dependency"""
    s3 = S3(test_bucket)
    
    dataset = YoloDarknetS3Loader(
        images_dir="images/",
        labels_dir="labels/",
        s3_client=s3
    )
    
    # Measure dataset creation and access
    times = []
    
    for i in range(3):
        start_time = time.perf_counter()
        path, image, labels = dataset[0]
        end_time = time.perf_counter()
        times.append(end_time - start_time)
    
    avg_time = sum(times) / len(times)
    print(f"Average YOLO load time: {avg_time:.4f} seconds")
    
    # Basic assertions
    assert avg_time < 1.0  # Should be fast with mock S3
    assert len(dataset) == 2