import unittest
import requests
import json
from typing import Dict, List
import time
from src.preprocessing.preprocessing import create_ordered_medical_pipeline


class TestPredictionAPI(unittest.TestCase):
    """Test suite for the Prediction API with detailed results"""

    BASE_URL = "http://localhost:5000"

    # Sample texts for each disease category based on EDA
    TEST_CASES = {
        "Dementia": {
            "case_1": {
                "description": "Dementia patient showing progressive cognitive decline with memory loss. MMSE score: 22. Difficulties with daily activities and confusion about time and place.",
                "expected_label": "Dementia"
            },
            "case_2": {
                "description": "this is a test description about Dementia",
                "expected_label": "Dementia"
            }
        },
        "ALS": {
            "case_1": {
                "description": "ALS Patient presents with progressive muscle weakness. FVC at 65% with declining respiratory function. ALSFRS-R score: 38.",
                "expected_label": "ALS"
            },
            "case_2": {
                "description": "Bulbar onset ALS with respiratory complications. FVC measurements show decline. ALSFRS-R decreased from 42 to 35.",
                "expected_label": "ALS"
            }
        },
        "Obsessive Compulsive Disorder": {
            "case_1": {
                "description": "Obsessive Compulsive Disorder Patient exhibits ritualistic behaviors and intrusive thoughts. Y-BOCS score: 28. Significant impact on daily functioning.",
                "expected_label": "Obsessive Compulsive Disorder"
            },
            "case_2": {
                "description": "Severe Obsessive Compulsive Disorder symptoms with compulsive checking behavior. Y-BOCS assessment indicates severe symptoms at 26.",
                "expected_label": "Obsessive Compulsive Disorder"
            }
        },
        "Scoliosis": {
            "case_1": {
                "description": "Scoliosis adolescent with spinal curvature. Cobb angle measurement of 45 degrees. Surgical correction being considered.",
                "expected_label": "Scoliosis"
            },
            "case_2": {
                "description": "Scoliosis progressive thoracic curve with 38-degree Cobb angle. Brace treatment initiated.",
                "expected_label": "Scoliosis"
            }
        },
        "Parkinson's Disease": {
            "case_1": {
                "description": "Parkinson Patient presents with resting tremor and bradykinesia. UPDRS score: 45. Started on levodopa therapy.",
                "expected_label": "Parkinson's Disease"
            },
            "case_2": {
                "description": "Progressive Parkinson symptoms with rigidity. Hoehn and Yahr stage 2. Motor symptoms predominant.",
                "expected_label": "Parkinson's Disease"
            }
        }
    }

    def setUp(self):
        """Set up test case"""
        try:
            response = requests.get(f"{self.BASE_URL}/health")
            response.raise_for_status()
            print("\nAPI Health Check: OK")
        except requests.exceptions.RequestException as e:
            raise unittest.SkipTest(f"API is not running: {str(e)}")

    def test_predictions_and_generate_report(self):
        """Test predictions for all classes and generate detailed JSON report"""

        results = {
            "test_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_cases": 0,
            "successful_predictions": 0,
            "failed_predictions": 0,
            "results_by_disease": {},
            "summary": {
                "accuracy_by_disease": {}
            }
        }

        # Test each disease category
        for disease, cases in self.TEST_CASES.items():
            disease_results = {
                "total_cases": len(cases),
                "successful_predictions": 0,
                "predictions": []
            }

            # Test each case for the disease
            for case_name, case_data in cases.items():
                print(f"\nTesting {disease} - {case_name}")

                # Make prediction request
                try:
                    response = requests.post(
                        f"{self.BASE_URL}/predict",
                        json={"description": case_data["description"]}
                    )

                    prediction_result = {
                        "case": case_name,
                        "expected_label": case_data["expected_label"],
                        "description": case_data["description"],
                        "api_response": {
                            "status_code": response.status_code,
                            "response_time": response.elapsed.total_seconds()
                        }
                    }

                    if response.status_code == 200:
                        response_data = response.json()
                        prediction_result["prediction"] = response_data["prediction"]
                        prediction_result["confidence"] = response_data.get("confidence")
                        prediction_result["correct"] = (
                                response_data["prediction"] == case_data["expected_label"]
                        )

                        if prediction_result["correct"]:
                            disease_results["successful_predictions"] += 1
                            results["successful_predictions"] += 1
                    else:
                        prediction_result["error"] = str(response.text)
                        results["failed_predictions"] += 1

                    disease_results["predictions"].append(prediction_result)
                    results["total_cases"] += 1

                except Exception as e:
                    print(f"Error processing case {case_name}: {str(e)}")
                    results["failed_predictions"] += 1

            # Calculate accuracy for this disease
            if disease_results["total_cases"] > 0:
                accuracy = (
                        disease_results["successful_predictions"] /
                        disease_results["total_cases"]
                )
                results["summary"]["accuracy_by_disease"][disease] = accuracy

            results["results_by_disease"][disease] = disease_results

        # Calculate overall accuracy
        total_successful = results["successful_predictions"]
        total_cases = results["total_cases"]
        results["summary"]["overall_accuracy"] = (
            total_successful / total_cases if total_cases > 0 else 0
        )

        # Save results to file
        with open('prediction_test_results.json', 'w') as f:
            json.dump(results, f, indent=2)

        print("\nTest Results Summary:")
        print(f"Total Cases: {results['total_cases']}")
        print(f"Successful Predictions: {results['successful_predictions']}")
        print(f"Failed Predictions: {results['failed_predictions']}")
        print("\nAccuracy by Disease:")
        for disease, accuracy in results["summary"]["accuracy_by_disease"].items():
            print(f"{disease}: {accuracy:.2%}")
        print(f"\nOverall Accuracy: {results['summary']['overall_accuracy']:.2%}")
        print("\nDetailed results saved to prediction_test_results.json")


if __name__ == '__main__':
    unittest.main(verbosity=2)
