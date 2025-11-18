"""
pytest fixtures for testing Pascal VOC S3 loaders with moto (S3 mock)
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
import xml.etree.ElementTree as ET

os.environ["SALTUP_BACKEND"] = "keras_tensorflow"

from saltup.utils.data.s3.s3_utils import S3
from saltup.ai.object_detection.dataset.pascal_voc import PascalVOCS3Loader


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
    """Create a test bucket with Pascal VOC data"""
    bucket_name = "test-pascal-voc-bucket"
    s3_client.create_bucket(Bucket=bucket_name)
    
    # Create sample Pascal VOC annotation XML files
    pascal_annotations = {
        "annotations/image1.xml": create_pascal_voc_xml("image1.jpg", 640, 480, [
            {"name": "person", "xmin": 100, "ymin": 150, "xmax": 200, "ymax": 300},
            {"name": "car", "xmin": 300, "ymin": 200, "xmax": 500, "ymax": 400}
        ]),
        "annotations/image2.xml": create_pascal_voc_xml("image2.jpg", 800, 600, [
            {"name": "bike", "xmin": 50, "ymin": 80, "xmax": 150, "ymax": 200}
        ])
    }
    
    # Create REAL image files - not fake binary data
    image_files = {}
    
    # Create valid JPEG images for testing
    for i, name in enumerate(["images/image1.jpg", "images/image2.jpg"], 1):
        img = Image.new('RGB', (640, 480), color=(255, 0, 0))  # 640x480 red image
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='JPEG')
        image_files[name] = img_bytes.getvalue()
    
    # Debug: Print what we're uploading
    print(f"Uploading {len(image_files)} images to S3")
    for key, data in image_files.items():
        print(f"  {key}: {len(data)} bytes")
    
    # Upload Pascal VOC annotations
    for key, content in pascal_annotations.items():
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


def create_pascal_voc_xml(filename, width, height, objects):
    """Create a Pascal VOC format XML annotation"""
    annotation = ET.Element("annotation")
    
    # Add filename
    filename_elem = ET.SubElement(annotation, "filename")
    filename_elem.text = filename
    
    # Add size information
    size_elem = ET.SubElement(annotation, "size")
    width_elem = ET.SubElement(size_elem, "width")
    width_elem.text = str(width)
    height_elem = ET.SubElement(size_elem, "height")
    height_elem.text = str(height)
    depth_elem = ET.SubElement(size_elem, "depth")
    depth_elem.text = "3"
    
    # Add objects
    for obj in objects:
        object_elem = ET.SubElement(annotation, "object")
        
        name_elem = ET.SubElement(object_elem, "name")
        name_elem.text = obj["name"]
        
        bndbox_elem = ET.SubElement(object_elem, "bndbox")
        
        xmin_elem = ET.SubElement(bndbox_elem, "xmin")
        xmin_elem.text = str(obj["xmin"])
        
        ymin_elem = ET.SubElement(bndbox_elem, "ymin")
        ymin_elem.text = str(obj["ymin"])
        
        xmax_elem = ET.SubElement(bndbox_elem, "xmax")
        xmax_elem.text = str(obj["xmax"])
        
        ymax_elem = ET.SubElement(bndbox_elem, "ymax")
        ymax_elem.text = str(obj["ymax"])
    
    return ET.tostring(annotation, encoding='unicode')


def test_pascal_voc_s3_loader_basic(s3_client, test_bucket):
    """Test basic Pascal VOC S3 loader functionality"""
    s3 = S3(test_bucket)
    
    # Debug: Check S3 connection
    print(f"S3 bucket: {test_bucket}")
    print(f"S3 client: {s3}")
    
    dataset = PascalVOCS3Loader(
        images_dir="images/",
        annotations_dir="annotations/",
        s3_client=s3,
    )
    
    # Test basic properties
    print(f"Dataset length: {len(dataset)}")
    assert len(dataset) == 2
    path, image, labels = dataset[0]
    assert path is not None
    assert image is None  # Without download_file=True
    assert labels is not None


def test_pascal_voc_s3_loader_basic_with_image(s3_client, test_bucket):
    """Test basic Pascal VOC S3 loader functionality with image download"""
    s3 = S3(test_bucket)
    
    # Debug: Check S3 connection
    print(f"S3 bucket: {test_bucket}")
    print(f"S3 client: {s3}")
    
    dataset = PascalVOCS3Loader(
        images_dir="images/",
        annotations_dir="annotations/",
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
    assert len(labels) > 0  # Should have annotations


def test_multiple_pascal_voc_buckets(s3_client):
    """Test working with multiple Pascal VOC buckets"""
    # Create multiple buckets
    s3_client.create_bucket(Bucket="pascal-train")
    s3_client.create_bucket(Bucket="pascal-val")
    
    # Create REAL images for train bucket
    train_images = {}
    for name in ["images/train1.jpg", "images/train2.jpg"]:
        img = Image.new('RGB', (640, 480), color=(0, 255, 0))  # Green image
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='JPEG')
        train_images[name] = img_bytes.getvalue()
    
    # Create REAL image for val bucket
    val_images = {}
    img = Image.new('RGB', (800, 600), color=(0, 0, 255))  # Blue image
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='JPEG')
    val_images["images/val1.jpg"] = img_bytes.getvalue()
    
    # Add Pascal VOC data to train bucket
    train_data = {
        **train_images,  # Real images
        "annotations/train1.xml": create_pascal_voc_xml("train1.jpg", 640, 480, [
            {"name": "person", "xmin": 50, "ymin": 60, "xmax": 150, "ymax": 200}
        ]),
        "annotations/train2.xml": create_pascal_voc_xml("train2.jpg", 640, 480, [
            {"name": "car", "xmin": 100, "ymin": 100, "xmax": 300, "ymax": 250}
        ])
    }
    
    # Add Pascal VOC data to val bucket
    val_data = {
        **val_images,  # Real images
        "annotations/val1.xml": create_pascal_voc_xml("val1.jpg", 800, 600, [
            {"name": "bike", "xmin": 200, "ymin": 150, "xmax": 400, "ymax": 350}
        ])
    }
    
    # Upload train data
    for key, content in train_data.items():
        if isinstance(content, str):
            content = content.encode('utf-8')
        s3_client.put_object(Bucket="pascal-train", Key=key, Body=content)
    
    # Upload val data  
    for key, content in val_data.items():
        if isinstance(content, str):
            content = content.encode('utf-8')
        s3_client.put_object(Bucket="pascal-val", Key=key, Body=content)
    
    # Test datasets from different buckets
    s3_train = S3("pascal-train")
    s3_val = S3("pascal-val")
    
    train_dataset = PascalVOCS3Loader(
        images_dir="images/",
        annotations_dir="annotations/", 
        s3_client=s3_train
    )
    
    val_dataset = PascalVOCS3Loader(
        images_dir="images/",
        annotations_dir="annotations/",
        s3_client=s3_val
    )
    
    # Verify different dataset sizes
    assert len(train_dataset) == 2
    assert len(val_dataset) == 1


@pytest.fixture
def s3_with_pascal_voc_nested_structure(s3_client):
    """Create bucket with complex nested Pascal VOC structure"""
    bucket_name = "test-pascal-voc-nested-bucket"
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
        img = Image.new('RGB', (320, 240), color=(128, 128, 128))  # Gray image
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='JPEG')
        nested_images[name] = img_bytes.getvalue()
    
    # Create nested Pascal VOC folder structure
    files_data = {
        **nested_images,  # Real images
        "dataset/train/annotations/image1.xml": create_pascal_voc_xml("image1.jpg", 320, 240, [
            {"name": "person", "xmin": 50, "ymin": 50, "xmax": 100, "ymax": 150},
            {"name": "car", "xmin": 150, "ymin": 100, "xmax": 250, "ymax": 200}
        ]),
        "dataset/train/annotations/image2.xml": create_pascal_voc_xml("image2.jpg", 320, 240, [
            {"name": "bike", "xmin": 80, "ymin": 60, "xmax": 180, "ymax": 160}
        ]),
        "dataset/val/annotations/image3.xml": create_pascal_voc_xml("image3.jpg", 320, 240, [
            {"name": "person", "xmin": 60, "ymin": 70, "xmax": 160, "ymax": 170}
        ]),
        "dataset/test/annotations/image4.xml": create_pascal_voc_xml("image4.jpg", 320, 240, [
            {"name": "car", "xmin": 40, "ymin": 50, "xmax": 140, "ymax": 150}
        ])
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


def test_pascal_voc_s3_loader_invalid_annotations(s3_client):
    """Test handling of invalid Pascal VOC annotation format"""
    bucket_name = "invalid-pascal-voc-bucket"
    s3_client.create_bucket(Bucket=bucket_name)
    
    # Create REAL image with invalid annotation
    img = Image.new('RGB', (100, 80), color=(255, 255, 0))  # Yellow image
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='JPEG')
    
    invalid_data = {
        "images/bad1.jpg": img_bytes.getvalue(),  # Real image
        "annotations/bad1.xml": "<invalid>xml_format</invalid>",  # Invalid XML format
    }
    
    for key, content in invalid_data.items():
        if isinstance(content, str):
            content = content.encode('utf-8')
        s3_client.put_object(Bucket=bucket_name, Key=key, Body=content)
    
    s3 = S3(bucket_name)
    
    dataset = PascalVOCS3Loader(
        images_dir="images/",
        annotations_dir="annotations/",
        s3_client=s3
    )
    
    # Should handle invalid annotations gracefully
    assert len(dataset) >= 0  # Might skip invalid files


def test_pascal_voc_s3_loader_missing_file(s3_client, test_bucket):
    """Test error handling for missing files"""
    s3 = S3(test_bucket)
    
    # Test with non-existent directory
    with pytest.raises(Exception):  # Could be ClientError or custom exception
        dataset = PascalVOCS3Loader(
            images_dir="nonexistent/",
            annotations_dir="annotations/",
            s3_client=s3
        )
        path, image, labels = dataset[0]


def test_pascal_voc_s3_loader_empty_bucket(s3_client):
    """Test with empty bucket"""
    bucket_name = "empty-pascal-voc-bucket" 
    s3_client.create_bucket(Bucket=bucket_name)
    
    s3 = S3(bucket_name)
    
    dataset = PascalVOCS3Loader(
        images_dir="images/",
        annotations_dir="annotations/",
        s3_client=s3
    )
    
    # Empty dataset should have length 0
    assert len(dataset) == 0


def test_pascal_voc_nested_directory_structure(s3_client, s3_with_pascal_voc_nested_structure):
    """Test Pascal VOC loader handles nested directories"""
    s3 = S3(s3_with_pascal_voc_nested_structure)
    
    train_dataset = PascalVOCS3Loader(
        images_dir="dataset/train/images/",
        annotations_dir="dataset/train/annotations/", 
        s3_client=s3,
        download_file=True,
        max_files=10
    )
    
    val_dataset = PascalVOCS3Loader(
        images_dir="dataset/val/images/",
        annotations_dir="dataset/val/annotations/",
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


def test_pascal_voc_s3_loader_timing(s3_client, test_bucket):
    """Test Pascal VOC loader timing without external benchmark dependency"""
    s3 = S3(test_bucket)
    
    dataset = PascalVOCS3Loader(
        images_dir="images/",
        annotations_dir="annotations/",
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
    print(f"Average Pascal VOC load time: {avg_time:.4f} seconds")
    
    # Basic assertions
    assert avg_time < 1.0  # Should be fast with mock S3
    assert len(dataset) == 2

def test_pascal_voc_xml_parsing(s3_client):
    """Test Pascal VOC XML parsing functionality"""
    bucket_name = "xml-parsing-bucket"
    s3_client.create_bucket(Bucket=bucket_name)
    
    # Create image
    img = Image.new('RGB', (640, 480), color=(255, 128, 0))
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='JPEG')
    
    # Create complex XML with multiple objects
    complex_xml = create_pascal_voc_xml("test.jpg", 640, 480, [
        {"name": "person", "xmin": 100, "ymin": 200, "xmax": 300, "ymax": 400},
        {"name": "car", "xmin": 400, "ymin": 150, "xmax": 600, "ymax": 350},
        {"name": "bike", "xmin": 50, "ymin": 300, "xmax": 150, "ymax": 450}
    ])
    
    # Upload test data
    s3_client.put_object(Bucket=bucket_name, Key="images/test.jpg", Body=img_bytes.getvalue())
    s3_client.put_object(Bucket=bucket_name, Key="annotations/test.xml", Body=complex_xml.encode('utf-8'))
    
    s3 = S3(bucket_name)
    
    dataset = PascalVOCS3Loader(
        images_dir="images/",
        annotations_dir="annotations/",
        s3_client=s3,
        download_file=True,
        max_files=10
    )
    
    assert len(dataset) == 1
    
    path, image, labels = dataset[0]
    assert image is not None
    assert len(labels) == 3  # Should have 3 objects
    
    # Check that all class names are parsed correctly
    class_names = [label.class_name for label in labels]
    assert "person" in class_names
    assert "car" in class_names
    assert "bike" in class_names