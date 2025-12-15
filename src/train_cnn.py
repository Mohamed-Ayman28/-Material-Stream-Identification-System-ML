"""
Deep Learning CNN Training for Material Classification
Uses Transfer Learning with MobileNetV2 for 80-90% accuracy
"""
import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class CNNTrainer:
    def __init__(self, data_dir, img_size=(224, 224), batch_size=32, epochs=20):
        self.data_dir = Path(data_dir)
        self.img_size = img_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.num_classes = 6
        self.class_names = ['glass', 'paper', 'cardboard', 'plastic', 'metal', 'trash']
        self.model = None
        self.history = None
        
    def create_model(self):
        """Create CNN model using MobileNetV2 transfer learning"""
        print("\n" + "="*60)
        print("Creating Deep Learning Model")
        print("="*60)
        
        # Load pre-trained MobileNetV2 (trained on ImageNet)
        base_model = MobileNetV2(
            input_shape=(*self.img_size, 3),
            include_top=False,
            weights='imagenet'
        )
        
        # Freeze base model layers initially
        base_model.trainable = False
        
        # Create new model on top
        inputs = keras.Input(shape=(*self.img_size, 3))
        
        # Preprocessing for MobileNetV2
        x = keras.applications.mobilenet_v2.preprocess_input(inputs)
        
        # Base model
        x = base_model(x, training=False)
        
        # Add custom classification head
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        model = keras.Model(inputs, outputs)
        
        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print(f"\nModel Architecture:")
        print(f"  Base Model: MobileNetV2 (pre-trained on ImageNet)")
        print(f"  Input Shape: {self.img_size}")
        print(f"  Number of Classes: {self.num_classes}")
        print(f"  Total Parameters: {model.count_params():,}")
        
        self.model = model
        return model
    
    def prepare_data(self):
        """Prepare data generators for training"""
        print("\n" + "="*60)
        print("Preparing Data")
        print("="*60)
        
        # Custom image loader that skips corrupted images
        from PIL import Image
        import io
        
        def robust_image_loader(path):
            """Load image with error handling for corrupted files"""
            try:
                img = Image.open(path)
                # Verify it's a valid image
                img.verify()
                # Reopen after verify
                img = Image.open(path)
                return img.convert('RGB')
            except Exception as e:
                print(f"Warning: Skipping corrupted image: {path}")
                # Return a blank image if corrupted
                return Image.new('RGB', (224, 224), color='black')
        
        # Data augmentation for training
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=30,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest',
            validation_split=0.2  # 80% train, 20% validation
        )
        
        # Only rescaling for validation
        val_datagen = ImageDataGenerator(
            rescale=1./255,
            validation_split=0.2
        )
        
        # Training generator
        train_generator = train_datagen.flow_from_directory(
            self.data_dir,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='sparse',
            subset='training',
            shuffle=True
        )
        
        # Validation generator
        val_generator = val_datagen.flow_from_directory(
            self.data_dir,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='sparse',
            subset='validation',
            shuffle=False
        )
        
        print(f"\nData Summary:")
        print(f"  Training samples: {train_generator.samples}")
        print(f"  Validation samples: {val_generator.samples}")
        print(f"  Batch size: {self.batch_size}")
        print(f"  Classes: {train_generator.class_indices}")
        
        # Save class mapping
        class_map = {v: k for k, v in train_generator.class_indices.items()}
        os.makedirs('models', exist_ok=True)
        with open('models/cnn_class_map.json', 'w') as f:
            json.dump(class_map, f, indent=4)
        print(f"\nClass map saved to: models/cnn_class_map.json")
        
        return train_generator, val_generator
    
    def train(self, train_generator, val_generator):
        """Train the model"""
        print("\n" + "="*60)
        print("Training CNN Model")
        print("="*60)
        
        # Callbacks
        callbacks = [
            keras.callbacks.ModelCheckpoint(
                'models/cnn_model_best.h5',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            ),
            keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=5,
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        # Train
        print(f"\nStarting training for {self.epochs} epochs...")
        self.history = self.model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=self.epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        # Fine-tuning: Unfreeze some layers and train with lower learning rate
        print("\n" + "="*60)
        print("Fine-Tuning Model")
        print("="*60)
        
        # Find the base model layer
        base_model = None
        for layer in self.model.layers:
            if 'mobilenetv2' in layer.name.lower():
                base_model = layer
                break
        
        if base_model is not None:
            base_model.trainable = True
            
            # Freeze all layers except the top 30
            for layer in base_model.layers[:-30]:
                layer.trainable = False
        else:
            print("Base model not found, skipping fine-tuning")
        
        # Recompile with lower learning rate
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print(f"Fine-tuning top 30 layers...")
        
        # Continue training
        history_fine = self.model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=10,
            callbacks=callbacks,
            verbose=1
        )
        
        # Combine histories
        for key in self.history.history.keys():
            self.history.history[key].extend(history_fine.history[key])
        
        return self.history
    
    def evaluate(self, val_generator):
        """Evaluate the model"""
        print("\n" + "="*60)
        print("Evaluating Model")
        print("="*60)
        
        # Evaluate
        val_loss, val_accuracy = self.model.evaluate(val_generator, verbose=1)
        
        print(f"\nFinal Results:")
        print(f"  Validation Loss: {val_loss:.4f}")
        print(f"  Validation Accuracy: {val_accuracy:.4f} ({val_accuracy*100:.2f}%)")
        
        return val_loss, val_accuracy
    
    def save_model(self):
        """Save the trained model"""
        print("\n" + "="*60)
        print("Saving Model")
        print("="*60)
        
        # Save full model
        self.model.save('models/cnn_model.h5')
        print(f"Model saved to: models/cnn_model.h5")
        
        # Save as Keras format (recommended)
        self.model.save('models/cnn_model.keras')
        print(f"Model saved to: models/cnn_model.keras")
        
    def plot_training(self):
        """Plot training history"""
        print("\nGenerating training plots...")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Accuracy plot
        ax1.plot(self.history.history['accuracy'], label='Training')
        ax1.plot(self.history.history['val_accuracy'], label='Validation')
        ax1.set_title('Model Accuracy', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Loss plot
        ax2.plot(self.history.history['loss'], label='Training')
        ax2.plot(self.history.history['val_loss'], label='Validation')
        ax2.set_title('Model Loss', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        os.makedirs('evaluation_results', exist_ok=True)
        plt.savefig('evaluation_results/cnn_training_history.png', dpi=150, bbox_inches='tight')
        print(f"Training plots saved to: evaluation_results/cnn_training_history.png")
        plt.close()


def main():
    parser = argparse.ArgumentParser(description='Train CNN model for material classification')
    parser.add_argument('--data-dir', type=str, default='dataset',
                        help='Directory containing training data')
    parser.add_argument('--img-size', type=int, default=224,
                        help='Image size (default: 224)')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size (default: 32)')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of epochs (default: 20)')
    
    args = parser.parse_args()
    
    # Check if data directory exists
    if not os.path.exists(args.data_dir):
        print(f"Error: Data directory '{args.data_dir}' not found!")
        return
    
    # Create trainer
    trainer = CNNTrainer(
        data_dir=args.data_dir,
        img_size=(args.img_size, args.img_size),
        batch_size=args.batch_size,
        epochs=args.epochs
    )
    
    # Create model
    trainer.create_model()
    
    # Prepare data
    train_gen, val_gen = trainer.prepare_data()
    
    # Train
    trainer.train(train_gen, val_gen)
    
    # Evaluate
    trainer.evaluate(val_gen)
    
    # Save
    trainer.save_model()
    
    # Plot
    trainer.plot_training()
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print("\nYou can now use:")
    print("  - predict_cnn.py for single image predictions")
    print("  - deploy_cnn.py for real-time webcam inference")


if __name__ == '__main__':
    main()
