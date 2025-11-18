"""
pytest fixtures for testing S3 loaders with moto (S3 mock)
"""
import pytest
import boto3
from moto import mock_aws
import json
import os
os.environ["SALTUP_BACKEND"] = "keras_tensorflow"
import json
import shutil
import numpy as np
from PIL import Image
from collections import defaultdict

from saltup.utils.data.s3.s3_utils import S3
from saltup.utils.data.image.image_utils import Image as SaltupImage
from saltup.ai.object_detection.utils.bbox import BBox, BBoxClassId
from saltup.ai.object_detection.dataset.coco import (
    create_dataset_structure, validate_dataset_structure,
    get_dataset_paths, read_annotations, write_annotations,
    replace_annotations_class, shift_class_ids,
    analyze_dataset, convert_coco_to_yolo_labels,
    split_dataset, split_and_organize_dataset,
    count_annotations, COCOS3Loader, ColorMode,
    is_coco_dataset
)


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
    """Create a test bucket with sample data"""
    bucket_name = "test-coco-bucket"
    s3_client.create_bucket(Bucket=bucket_name)
    
    # Upload sample COCO annotation file
    coco_annotations = {
        "images": [
            {"id": 1, "file_name": "image1.jpg", "width": 640, "height": 480},
            {"id": 2, "file_name": "image2.jpg", "width": 800, "height": 600}
        ],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 20, 100, 150]},
            {"id": 2, "image_id": 2, "category_id": 2, "bbox": [50, 60, 200, 250]}
        ],
        "categories": [
            {"id": 1, "name": "cat"},
            {"id": 2, "name": "dog"}
        ]
    }
    
    s3_client.put_object(
        Bucket=bucket_name,
        Key="annotations/instances.json",
        Body=json.dumps(coco_annotations)
    )
    
    # Upload dummy images
    s3_client.put_object(
        Bucket=bucket_name,
        Key="images/image1.jpg",
        Body=b"fake_image_data_1"
    )
    s3_client.put_object(
        Bucket=bucket_name,
        Key="images/image2.jpg",
        Body=b"fake_image_data_2"
    )
    
    yield bucket_name
    # No cleanup needed - mock is destroyed after test


# Example test
@mock_aws
def test_coco_s3_loader(aws_credentials):
    """Test CocoS3Loader with moto backend"""
    # Setup within the test (alternative approach)
    s3 = boto3.client('s3', region_name='us-east-1')
    bucket_name = "test-coco-bucket"
    s3.create_bucket(Bucket=bucket_name)
    
    # Upload test data
    coco_annotations = {
        "images": [
            {"id": 1, "file_name": "image1.jpg", "width": 640, "height": 480},
            {"id": 2, "file_name": "image2.jpg", "width": 800, "height": 600}
        ],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 20, 100, 150]},
            {"id": 2, "image_id": 2, "category_id": 2, "bbox": [50, 60, 200, 250]}
        ],
        "categories": [
            {"id": 1, "name": "cat"},
            {"id": 2, "name": "dog"}
        ]
    }
    s3.put_object(
        Bucket=bucket_name,
        Key="annotations/instances.json",
        Body=json.dumps(coco_annotations)
    ) 
    s3_client = S3(bucket_name)
    loader = COCOS3Loader(
        images_dir="images/",
        annotations_file="annotations/instances.json",
        s3_client=s3_client
        # No endpoint_url needed - moto intercepts boto3 calls automatically
    )
    
    # Test loading annotations
    annotations = loader.annotations
    assert len(annotations['images']) == 2
    assert len(annotations['annotations']) == 2
    assert len(annotations['categories']) == 2


def test_coco_s3_loader_with_fixtures(s3_client, test_bucket):
    """Test CocoS3Loader using fixtures (cleaner approach)"""
    s3 = S3(test_bucket)
    # moto automatically intercepts boto3 calls - no special config needed
    loader = COCOS3Loader(
        images_dir="images/",
        annotations_file="annotations/instances.json",
        s3_client=s3
        # No endpoint_url needed - moto intercepts boto3 calls automatically
    )
    
    annotations = loader.annotations
    assert len(annotations['images']) == 2
    assert annotations['images'][0]['file_name'] == "image1.jpg"


def test_coco_s3_loader_missing_file(s3_client, test_bucket):
    """Test error handling for missing files"""
    from botocore.exceptions import ClientError
    
    s3 = S3(test_bucket)
    
    # The exception should be raised during loader instantiation, not when accessing .annotations
    with pytest.raises(ClientError) as exc_info:
        loader = COCOS3Loader(
            images_dir="images/",
            annotations_file="annotations/nonexistent.json",
            s3_client=s3
        )
    
    # Check that it's specifically a NoSuchKey error
    assert exc_info.value.response['Error']['Code'] == 'NoSuchKey'

def test_multiple_buckets(s3_client):
    """Test working with multiple buckets"""
    # Create multiple buckets
    s3_client.create_bucket(Bucket="bucket-train")
    s3_client.create_bucket(Bucket="bucket-val")
    
    # Add valid COCO format data to each bucket
    train_coco_data = {
        "images": [
            {"id": 1, "file_name": "train1.jpg", "width": 640, "height": 480}
        ],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 20, 100, 150]}
        ],
        "categories": [
            {"id": 1, "name": "person"}
        ]
    }
    
    val_coco_data = {
        "images": [
            {"id": 1, "file_name": "val1.jpg", "width": 800, "height": 600}
        ],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [50, 60, 200, 250]}
        ],
        "categories": [
            {"id": 1, "name": "car"}
        ]
    }
    
    s3_client.put_object(
        Bucket="bucket-train", 
        Key="data.json", 
        Body=json.dumps(train_coco_data)
    )
    s3_client.put_object(
        Bucket="bucket-val", 
        Key="data.json", 
        Body=json.dumps(val_coco_data)
    )
    
    # Test your loader with different buckets
    s3_train = S3("bucket-train")
    s3_val = S3("bucket-val")
    train_loader = COCOS3Loader(
        images_dir="images/", 
        annotations_file="data.json", 
        s3_client=s3_train
    )
    val_loader = COCOS3Loader(
        images_dir="images/", 
        annotations_file="data.json", 
        s3_client=s3_val
    )
    
    train_data = train_loader.annotations
    val_data = val_loader.annotations
    
    # Test COCO format validation
    assert 'images' in train_data
    assert 'annotations' in train_data
    assert 'categories' in train_data
    assert 'images' in val_data
    assert 'annotations' in val_data
    assert 'categories' in val_data
    
    # Test specific content differences
    assert train_data['images'][0]['file_name'] == "train1.jpg"
    assert val_data['images'][0]['file_name'] == "val1.jpg"
    assert train_data['categories'][0]['name'] == "person"
    assert val_data['categories'][0]['name'] == "car"


@pytest.fixture
def s3_with_nested_structure(s3_client):
    """Create bucket with complex nested structure"""
    bucket_name = "test-nested-bucket"
    s3_client.create_bucket(Bucket=bucket_name)
    
    # Create COCO annotations for train set
    train_annotations = {
        "images": [
            {"id": 1, "file_name": "image1.jpg", "width": 640, "height": 480},
            {"id": 2, "file_name": "image2.jpg", "width": 800, "height": 600}
        ],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 20, 100, 150]},
            {"id": 2, "image_id": 2, "category_id": 2, "bbox": [30, 40, 200, 300]}
        ],
        "categories": [
            {"id": 1, "name": "person"},
            {"id": 2, "name": "car"}
        ]
    }
    
    # Create COCO annotations for val set
    val_annotations = {
        "images": [
            {"id": 3, "file_name": "image3.jpg", "width": 1024, "height": 768}
        ],
        "annotations": [
            {"id": 3, "image_id": 3, "category_id": 1, "bbox": [50, 60, 150, 200]}
        ],
        "categories": [
            {"id": 1, "name": "person"}
        ]
    }
    
    # Create nested folder structure with proper data
    files_data = {
        "dataset/train/images/image1.jpg": b"fake_image_data_1",
        "dataset/train/images/image2.jpg": b"fake_image_data_2",
        "dataset/train/annotations.json": json.dumps(train_annotations).encode('utf-8'),
        "dataset/val/images/image3.jpg": b"fake_image_data_3",
        "dataset/val/annotations.json": json.dumps(val_annotations).encode('utf-8'),
        "dataset/test/images/image4.jpg": b"fake_image_data_4",
    }
    
    for path, data in files_data.items():
        s3_client.put_object(
            Bucket=bucket_name,
            Key=path,
            Body=data
        )
    
    yield bucket_name


def test_nested_directory_structure(s3_client, s3_with_nested_structure):
    """Test loader handles nested directories"""
    s3 = S3(s3_with_nested_structure)
    loader = COCOS3Loader(
        images_dir="dataset/train/images/", 
        annotations_file="dataset/train/annotations.json", 
        s3_client=s3
    )
    
    annotations = loader.annotations
    
    # Test COCO format structure
    assert 'images' in annotations
    assert 'annotations' in annotations
    assert 'categories' in annotations
    
    # Test content
    assert len(annotations['images']) == 2
    assert len(annotations['annotations']) == 2
    assert len(annotations['categories']) == 2
    
    # Test image data
    image_files = [img['file_name'] for img in annotations['images']]
    assert 'image1.jpg' in image_files
    assert 'image2.jpg' in image_files
    
    # Test categories
    category_names = [cat['name'] for cat in annotations['categories']]
    assert 'person' in category_names
    assert 'car' in category_names


def test_nested_val_directory(s3_client, s3_with_nested_structure):
    """Test val directory structure"""
    s3 = S3(s3_with_nested_structure)
    val_loader = COCOS3Loader(
        images_dir="dataset/val/images/", 
        annotations_file="dataset/val/annotations.json", 
        s3_client=s3
    )
    
    annotations = val_loader.annotations
    
    # Test val set has different structure
    assert len(annotations['images']) == 1
    assert len(annotations['annotations']) == 1
    assert annotations['images'][0]['file_name'] == 'image3.jpg'
    assert annotations['categories'][0]['name'] == 'person'


# Performance test example
def test_loader_performance(s3_client, test_bucket):
    """Test loader performance without benchmark dependency"""
    
    s3 = S3(test_bucket)
    loader = COCOS3Loader(
        images_dir="images/",
        annotations_file="annotations/instances.json",
        s3_client=s3
        # No endpoint_url needed - moto intercepts boto3 calls automatically
    )
    
    # Test the load operation
    import time
    start_time = time.time()
    result = loader.annotations
    end_time = time.time()
    
    # Basic performance assertion (should load quickly with mock data)
    load_time = end_time - start_time
    assert load_time < 1.0  # Should load in less than 1 second
    
    # Verify the data loaded correctly
    assert len(result['images']) == 2
    assert len(result['annotations']) == 2
    assert len(result['categories']) == 2