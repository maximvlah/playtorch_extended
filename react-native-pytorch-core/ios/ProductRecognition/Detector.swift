//
//  Detector.swift
//  PyTorchCore
//
//  Created by Maksim Vlah on 08.09.22.
//  Copyright © 2022 Facebook. All rights reserved.
//

import Foundation
import Vision

// MARK: - Detector
class Detector {
  
  /// CoreML model instance
  internal var model: VNCoreMLModel?
  
  /// Model filename & extension
  internal let modelFileName: String = "best"
  internal let modelFileExtension: String = "mlmodelc"
  
  /// Detection confidence threshold
  internal let CONFIDENCE_THRESHOLD: Float = 0.72
  
  /// Maximum number of detections to process
  internal let MAX_DETECTIONS: Int = 1
  
  init() {
    loadModel()
  }
  
  // Loads detector CoreML model.
  private func loadModel() -> NSError? {
      print("Loading product detector...")
      let error: NSError! = nil
        
      guard let modelURL = Bundle.main.url(forResource: modelFileName, withExtension: modelFileExtension) else {
          return NSError(domain: "Detector", code: -1, userInfo: [NSLocalizedDescriptionKey: "Model file is missing"])
      }
      
      do {
        model = try VNCoreMLModel(for: MLModel(contentsOf: modelURL))
        print("Loaded product detector!")

      } catch let error as NSError {
          print("Model loading went wrong: \(error)")
      }
      
      return error
  }
  
  // Detect products in the frame.
  func detect(_ pixelBuffer: CVPixelBuffer) -> [VNRecognizedObjectObservation] {
    
    var observations: [VNRecognizedObjectObservation] = []
    
    let request = VNCoreMLRequest(model: model!) { (finishedReq, err) in
          
      print("Detecting...")
      guard let detections = finishedReq.results as? [VNRecognizedObjectObservation] else { return }
      
      print("Detected \(detections.count) products.")
      if detections.count > 0 {
        
        var _detectionCounter = 0 // only output limited amount of detections
        for detection in detections where detection.confidence >= self.CONFIDENCE_THRESHOLD {
          
          observations.append(detection)
          
          _detectionCounter += 1
          if _detectionCounter == self.MAX_DETECTIONS { break }
        }
      }

    }
          
    try? VNImageRequestHandler(cvPixelBuffer: pixelBuffer, options: [:]).perform([request])
    
    return observations

  }
}

