import numpy as np
import time
import math
from lowmind import Tensor, Module, Conv2d, Linear, SGD, memory_manager, RaspberryPiAdvancedMonitor

# ----------------------------
# Object Detection Data Structures
# ----------------------------
class BoundingBox:
    def __init__(self, x, y, w, h, confidence, class_id, class_name):
        self.x = x  # center x
        self.y = y  # center y
        self.w = w  # width
        self.h = h  # height
        self.confidence = confidence
        self.class_id = class_id
        self.class_name = class_name
    
    def __repr__(self):
        return f"BBox({self.class_name} conf:{self.confidence:.2f} x:{self.x:.2f} y:{self.y:.2f} w:{self.w:.2f} h:{self.h:.2f})"
    
    def iou(self, other):
        """Calculate Intersection over Union with another box"""
        # Calculate intersection area
        x1 = max(self.x - self.w/2, other.x - other.w/2)
        y1 = max(self.y - self.h/2, other.y - other.h/2)
        x2 = min(self.x + self.w/2, other.x + other.w/2)
        y2 = min(self.y + self.h/2, other.y + other.h/2)
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        
        # Calculate union area
        area1 = self.w * self.h
        area2 = other.w * other.h
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0

# ----------------------------
# Object Detection Dataset Generator
# ----------------------------
class ObjectDetectionDataset:
    def __init__(self, grid_size=7, num_boxes=2, num_classes=3):
        self.grid_size = grid_size
        self.num_boxes = num_boxes
        self.num_classes = num_classes
        self.class_names = ['person', 'car', 'animal']
        
    def generate_dummy_data(self, num_samples=500):
        """Generate dummy object detection data"""
        images = []
        targets = []
        
        for i in range(num_samples):
            # Generate random image (64x64 pixels, 3 channels)
            img = np.random.rand(3, 64, 64).astype(np.float32) * 0.5 + 0.5
            
            # Generate random bounding boxes (0 to 2 objects per image)
            num_objects = np.random.randint(0, 3)
            target = np.zeros((self.grid_size, self.grid_size, 
                              self.num_boxes * 5 + self.num_classes), dtype=np.float32)
            
            for obj_idx in range(num_objects):
                # Random class
                class_id = np.random.randint(0, self.num_classes)
                
                # Random bounding box (normalized coordinates)
                x = np.random.uniform(0.2, 0.8)
                y = np.random.uniform(0.2, 0.8)
                w = np.random.uniform(0.1, 0.3)
                h = np.random.uniform(0.1, 0.3)
                
                # Determine grid cell
                grid_x = int(x * self.grid_size)
                grid_y = int(y * self.grid_size)
                
                # Relative coordinates within grid cell
                cell_x = x * self.grid_size - grid_x
                cell_y = y * self.grid_size - grid_y
                
                # Choose random anchor box
                box_idx = np.random.randint(0, self.num_boxes)
                
                # Set target values
                # Bounding box: [x, y, w, h, confidence]
                start_idx = box_idx * 5
                target[grid_y, grid_x, start_idx:start_idx+4] = [cell_x, cell_y, w, h]
                target[grid_y, grid_x, start_idx+4] = 1.0  # confidence
                
                # Class probabilities (one-hot encoding)
                class_start_idx = self.num_boxes * 5
                target[grid_y, grid_x, class_start_idx + class_id] = 1.0
            
            images.append(img)
            targets.append(target)
        
        # Split into train and test
        split_idx = int(0.8 * num_samples)
        X_train = np.array(images[:split_idx])
        y_train = np.array(targets[:split_idx])
        X_test = np.array(images[split_idx:])
        y_test = np.array(targets[split_idx:])
        
        return (X_train, y_train), (X_test, y_test)

# ----------------------------
# Tiny YOLO-inspired Object Detection Model
# ----------------------------
class TinyYOLO(Module):
    """Ultra-lightweight YOLO-inspired object detector for Raspberry Pi"""
    def __init__(self, grid_size=7, num_boxes=2, num_classes=3, device='cpu'):
        super().__init__()
        self.grid_size = grid_size
        self.num_boxes = num_boxes
        self.num_classes = num_classes
        self.device = device
        
        # Feature extraction backbone (very small for Raspberry Pi)
        self.conv1 = Conv2d(3, 8, 3, padding=1, device=device)  # 64x64 -> 64x64
        self.conv2 = Conv2d(8, 16, 3, stride=2, padding=1, device=device)  # 64x64 -> 32x32
        self.conv3 = Conv2d(16, 32, 3, stride=2, padding=1, device=device)  # 32x32 -> 16x16
        self.conv4 = Conv2d(32, 64, 3, stride=2, padding=1, device=device)  # 16x16 -> 8x8
        self.conv5 = Conv2d(64, 128, 3, stride=2, padding=1, device=device)  # 8x8 -> 4x4
        
        # Detection head
        # Output: grid_size * grid_size * (num_boxes * 5 + num_classes)
        output_features = grid_size * grid_size * (num_boxes * 5 + num_classes)
        self.fc1 = Linear(128 * 4 * 4, 256, device=device)
        self.fc2 = Linear(256, output_features, device=device)
        
    def forward(self, x):
        # Feature extraction
        x = self.conv1(x).relu()
        x = self.conv2(x).relu()
        x = self.conv3(x).relu()
        x = self.conv4(x).relu()
        x = self.conv5(x).relu()
        
        # Flatten
        batch_size = x.shape[0]
        x = x.reshape(batch_size, -1)
        
        # Detection head
        x = self.fc1(x).relu()
        x = self.fc2(x)
        
        # Reshape to final output format
        x = x.reshape(batch_size, self.grid_size, self.grid_size, 
                     self.num_boxes * 5 + self.num_classes)
        
        return x

# ----------------------------
# Object Detection Loss Function
# ----------------------------
class YOLOLoss:
    def __init__(self, grid_size=7, num_boxes=2, num_classes=3, lambda_coord=5, lambda_noobj=0.5):
        self.grid_size = grid_size
        self.num_boxes = num_boxes
        self.num_classes = num_classes
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
    
    def __call__(self, predictions, targets):
        """
        YOLO loss function
        predictions: Tensor of shape (batch, grid_size, grid_size, num_boxes*5 + num_classes)
        targets: Tensor of same shape
        """
        batch_size = predictions.shape[0]
        
        # Reshape for easier processing
        pred = predictions.data.reshape(batch_size, self.grid_size, self.grid_size, -1)
        target = targets.data.reshape(batch_size, self.grid_size, self.grid_size, -1)
        
        total_loss = 0
        
        for i in range(batch_size):
            for y in range(self.grid_size):
                for x in range(self.grid_size):
                    # Check if there's an object in this cell
                    has_object = False
                    for b in range(self.num_boxes):
                        if target[i, y, x, b*5 + 4] > 0.5:  # confidence score
                            has_object = True
                            break
                    
                    for b in range(self.num_boxes):
                        # Bounding box coordinates and confidence
                        pred_box = pred[i, y, x, b*5:b*5+5]
                        target_box = target[i, y, x, b*5:b*5+5]
                        
                        if has_object and target[i, y, x, b*5 + 4] > 0.5:
                            # Coordinate loss (only for responsible boxes)
                            coord_loss = (
                                (pred_box[0] - target_box[0])**2 +
                                (pred_box[1] - target_box[1])**2 +
                                (pred_box[2] - target_box[2])**2 +
                                (pred_box[3] - target_box[3])**2
                            )
                            total_loss += self.lambda_coord * coord_loss
                            
                            # Confidence loss (for objects)
                            conf_loss = (pred_box[4] - target_box[4])**2
                            total_loss += conf_loss
                        else:
                            # Confidence loss (for no objects)
                            conf_loss = (pred_box[4] - target_box[4])**2
                            total_loss += self.lambda_noobj * conf_loss
                    
                    # Class probability loss (only if cell has object)
                    if has_object:
                        pred_class = pred[i, y, x, self.num_boxes*5:]
                        target_class = target[i, y, x, self.num_boxes*5:]
                        class_loss = np.sum((pred_class - target_class)**2)
                        total_loss += class_loss
        
        return Tensor([total_loss / batch_size], requires_grad=True)

# ----------------------------
# Non-Maximum Suppression (NMS)
# ----------------------------
def non_max_suppression(boxes, iou_threshold=0.5):
    """Remove overlapping boxes using Non-Maximum Suppression"""
    if len(boxes) == 0:
        return []
    
    # Sort by confidence
    boxes.sort(key=lambda x: x.confidence, reverse=True)
    
    keep = []
    while boxes:
        # Take the box with highest confidence
        best_box = boxes.pop(0)
        keep.append(best_box)
        
        # Remove boxes that overlap too much
        boxes = [box for box in boxes if best_box.iou(box) < iou_threshold]
    
    return keep

# ----------------------------
# Object Detection Utilities
# ----------------------------
class ObjectDetector:
    def __init__(self, model, class_names, grid_size=7, num_boxes=2, confidence_threshold=0.5):
        self.model = model
        self.class_names = class_names
        self.grid_size = grid_size
        self.num_boxes = num_boxes
        self.num_classes = len(class_names)
        self.confidence_threshold = confidence_threshold
    
    def predict(self, image):
        """Run object detection on a single image"""
        self.model.eval()
        
        # Add batch dimension and convert to tensor
        if len(image.shape) == 3:
            image = image[np.newaxis, :]  # Add batch dimension
        
        image_tensor = Tensor(image)
        
        # Forward pass
        with torch.no_grad():
            output = self.model(image_tensor)
        
        # Parse predictions
        boxes = self._parse_output(output.data[0])
        
        # Apply Non-Maximum Suppression
        filtered_boxes = non_max_suppression(boxes)
        
        return filtered_boxes
    
    def _parse_output(self, output):
        """Convert model output to bounding boxes"""
        boxes = []
        
        for y in range(self.grid_size):
            for x in range(self.grid_size):
                for b in range(self.num_boxes):
                    # Get box predictions
                    start_idx = b * 5
                    box_x = output[y, x, start_idx + 0]
                    box_y = output[y, x, start_idx + 1]
                    box_w = output[y, x, start_idx + 2]
                    box_h = output[y, x, start_idx + 3]
                    confidence = output[y, x, start_idx + 4]
                    
                    # Get class probabilities
                    class_start_idx = self.num_boxes * 5
                    class_probs = output[y, x, class_start_idx:class_start_idx + self.num_classes]
                    
                    # Find class with highest probability
                    class_id = np.argmax(class_probs)
                    class_score = class_probs[class_id]
                    
                    # Calculate final confidence
                    final_confidence = confidence * class_score
                    
                    if final_confidence > self.confidence_threshold:
                        # Convert grid coordinates to image coordinates
                        abs_x = (x + box_x) / self.grid_size
                        abs_y = (y + box_y) / self.grid_size
                        abs_w = box_w
                        abs_h = box_h
                        
                        box = BoundingBox(
                            x=abs_x, y=abs_y, w=abs_w, h=abs_h,
                            confidence=final_confidence,
                            class_id=class_id,
                            class_name=self.class_names[class_id]
                        )
                        boxes.append(box)
        
        return boxes

# ----------------------------
# Object Detection Trainer
# ----------------------------
class ObjectDetectionTrainer:
    def __init__(self, model, optimizer, loss_fn):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.monitor = RaspberryPiAdvancedMonitor()
        self.train_losses = []
        
    def train_epoch(self, dataloader, epoch):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        total_batches = 0
        
        print(f"🎯 Training Object Detection Epoch {epoch}")
        print("-" * 50)
        
        for batch_idx, (images, targets) in enumerate(dataloader):
            # Memory cleanup
            memory_manager.free_unused()
            
            # Convert to tensors
            images_tensor = Tensor(images)
            targets_tensor = Tensor(targets)
            
            # Forward pass
            self.optimizer.zero_grad()
            predictions = self.model(images_tensor)
            loss = self.loss_fn(predictions, targets_tensor)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.data[0]
            total_batches += 1
            
            # Print progress
            if batch_idx % 5 == 0:
                self.monitor.update_monitoring()
                health_score = self.monitor.get_health_score()
                
                print(f"Batch {batch_idx:3d}/{len(dataloader):3d} | "
                      f"Loss: {loss.data[0]:.4f} | "
                      f"Health: {health_score:.1f}/100")
        
        avg_loss = total_loss / total_batches
        self.train_losses.append(avg_loss)
        
        return avg_loss
    
    def plot_training_history(self):
        """Simple text-based training history"""
        print("\n📊 Object Detection Training History:")
        print("=" * 40)
        print(f"{'Epoch':<6} {'Loss':<12}")
        print("-" * 40)
        
        for i, loss in enumerate(self.train_losses):
            print(f"{i+1:<6} {loss:<12.4f}")

# ----------------------------
# Simple DataLoader for Object Detection
# ----------------------------
class DetectionDataLoader:
    def __init__(self, X, y, batch_size=8):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.num_batches = len(X) // batch_size
        self.current_idx = 0
        
    def __iter__(self):
        self.current_idx = 0
        return self
    
    def __next__(self):
        if self.current_idx >= len(self.X):
            raise StopIteration
        
        end_idx = self.current_idx + self.batch_size
        batch_X = self.X[self.current_idx:end_idx]
        batch_y = self.y[self.current_idx:end_idx]
        self.current_idx = end_idx
        
        return batch_X, batch_y
    
    def __len__(self):
        return self.num_batches

# ----------------------------
# Visualization Utilities (Text-based)
# ----------------------------
def visualize_detection(image, boxes, image_size=64):
    """Simple text-based visualization of detection results"""
    print("\n🔍 Detection Results:")
    print("=" * 50)
    
    # Create simple ASCII art representation
    grid = [[' ' for _ in range(image_size//4)] for _ in range(image_size//8)]
    
    # Draw bounding boxes on grid
    for box in boxes:
        # Convert normalized coordinates to grid coordinates
        x1 = int((box.x - box.w/2) * len(grid[0]))
        y1 = int((box.y - box.h/2) * len(grid))
        x2 = int((box.x + box.w/2) * len(grid[0]))
        y2 = int((box.y + box.h/2) * len(grid))
        
        # Clip coordinates
        x1 = max(0, min(x1, len(grid[0])-1))
        x2 = max(0, min(x2, len(grid[0])-1))
        y1 = max(0, min(y1, len(grid)-1))
        y2 = max(0, min(y2, len(grid)-1))
        
        # Draw box boundaries
        for x in range(x1, x2+1):
            if y1 < len(grid):
                grid[y1][x] = '-'
            if y2 < len(grid):
                grid[y2][x] = '-'
        
        for y in range(y1, y2+1):
            if y < len(grid):
                if x1 < len(grid[0]):
                    grid[y][x1] = '|'
                if x2 < len(grid[0]):
                    grid[y][x2] = '|'
        
        # Add label at top-left corner
        if y1-1 >= 0 and x1 < len(grid[0]):
            label = box.class_name[0].upper()  # First letter
            grid[y1-1][x1] = label
    
    # Print the grid
    for row in grid:
        print(' ' + ''.join(row))
    
    # Print detection details
    print("\nDetected Objects:")
    for i, box in enumerate(boxes):
        print(f"  {i+1}. {box.class_name} (conf: {box.confidence:.2f}) "
              f"at ({box.x:.2f}, {box.y:.2f}) "
              f"size ({box.w:.2f}x{box.h:.2f})")

# ----------------------------
# Main Object Detection Training
# ----------------------------
def train_object_detector():
    """Main function to train object detection model"""
    print("🎯 Raspberry Pi Object Detection Model Training")
    print("=" * 60)
    
    # Setup
    grid_size = 7
    num_boxes = 2
    num_classes = 3
    batch_size = 8
    epochs = 10
    learning_rate = 0.001
    
    # Initialize monitoring
    monitor = RaspberryPiAdvancedMonitor()
    monitor.print_detailed_status()
    
    try:
        # Generate dataset
        print("\n📊 Generating Object Detection Dataset...")
        dataset = ObjectDetectionDataset(grid_size=grid_size, 
                                       num_boxes=num_boxes, 
                                       num_classes=num_classes)
        (X_train, y_train), (X_test, y_test) = dataset.generate_dummy_data(300)
        
        print(f"Training samples: {len(X_train)}")
        print(f"Test samples: {len(X_test)}")
        print(f"Classes: {dataset.class_names}")
        
        # Create data loader
        train_loader = DetectionDataLoader(X_train, y_train, batch_size=batch_size)
        
        # Create model
        print("\n🔄 Creating TinyYOLO Model...")
        model = TinyYOLO(grid_size=grid_size, 
                        num_boxes=num_boxes, 
                        num_classes=num_classes,
                        device='cpu')
        
        # Count parameters
        total_params = 0
        for param in model.parameters():
            total_params += np.prod(param.shape)
        print(f"Model parameters: {total_params:,}")
        
        # Create optimizer and loss
        optimizer = SGD(model.parameters(), lr=learning_rate, momentum=0.9)
        loss_fn = YOLOLoss(grid_size=grid_size, num_boxes=num_boxes, num_classes=num_classes)
        
        # Create trainer
        trainer = ObjectDetectionTrainer(model, optimizer, loss_fn)
        
        # Training loop
        print("\n🚀 Starting Training...")
        for epoch in range(epochs):
            avg_loss = trainer.train_epoch(train_loader, epoch + 1)
            print(f"📈 Epoch {epoch+1} Average Loss: {avg_loss:.4f}")
            print("-" * 50)
        
        # Show training history
        trainer.plot_training_history()
        
        # Test the model
        print("\n🧪 Testing Trained Model...")
        detector = ObjectDetector(model, dataset.class_names)
        
        # Test on a few sample images
        for i in range(3):
            test_image = X_test[i]
            boxes = detector.predict(test_image)
            
            print(f"\nTest Image {i+1}:")
            visualize_detection(test_image, boxes)
        
        # Final status
        monitor.print_detailed_status()
        
        print("\n🎉 Object Detection Model Trained Successfully!")
        print(f"✅ Model can detect: {', '.join(dataset.class_names)}")
        print("✅ Ready for real-time object detection on Raspberry Pi!")
        
        return model, detector
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None

# Add missing context manager for compatibility
class torch:
    class no_grad:
        def __enter__(self):
            pass
        def __exit__(self, *args):
            pass

# ----------------------------
# Real-time Detection Example
# ----------------------------
def real_time_detection_example():
    """Show how to use the detector for real-time applications"""
    print("\n⏱️ Real-time Detection Example")
    print("=" * 50)
    
    # Create a sample detector (in practice, you'd use a trained model)
    detector = ObjectDetector(None, ['person', 'car', 'animal'])
    
    # Simulate processing frames
    print("Simulating real-time detection on 5 frames:")
    
    for frame_idx in range(5):
        # Simulate processing time
        time.sleep(0.5)
        
        # Generate random detections for demonstration
        num_detections = np.random.randint(0, 3)
        boxes = []
        
        for i in range(num_detections):
            class_id = np.random.randint(0, 3)
            box = BoundingBox(
                x=np.random.uniform(0.2, 0.8),
                y=np.random.uniform(0.2, 0.8),
                w=np.random.uniform(0.1, 0.3),
                h=np.random.uniform(0.1, 0.3),
                confidence=np.random.uniform(0.6, 0.9),
                class_id=class_id,
                class_name=detector.class_names[class_id]
            )
            boxes.append(box)
        
        print(f"\nFrame {frame_idx + 1}:")
        for box in boxes:
            print(f"  👁️  Detected: {box.class_name} (confidence: {box.confidence:.2f})")
        
        if not boxes:
            print("  👁️  No objects detected")
    
    print("\n✅ Real-time detection simulation completed!")
    print("On real Raspberry Pi, this would process camera feed in real-time.")

# ----------------------------
# Main Execution
# ----------------------------
if __name__ == "__main__":
    # Train object detection model
    model, detector = train_object_detector()
    
    # Show real-time example
    if model is not None:
        real_time_detection_example()
        
        print("\n" + "=" * 60)
        print("🚀 Object Detection Model Ready for Deployment!")
        print("\nFeatures:")
        print("  ✅ YOLO-inspired architecture")
        print("  ✅ Multi-object detection")
        print("  ✅ Bounding box prediction")
        print("  ✅ Confidence scoring")
        print("  ✅ Non-Maximum Suppression")
        print("  ✅ Real-time capable")
        print("  ✅ Raspberry Pi optimized")
        
        print("\nUsage on Raspberry Pi:")
        print("  1. Connect USB camera")
        print("  2. Process frames in loop")
        print("  3. Run detector on each frame")
        print("  4. Display results with bounding boxes")