# Import standard libraries
import os
import json
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta
import scipy.signal
import scipy.stats

# Try to import optional dependencies
try:
    from langchain.prompts import PromptTemplate
    from langchain_core.output_parsers import PydanticOutputParser
    from langchain_core.runnables import RunnableSequence
    # Use the newer HuggingFace endpoint instead of deprecated one
    try:
        from langchain_huggingface import HuggingFaceEndpoint
        HAS_NEW_HF = True
    except ImportError:
        from langchain_community.llms import HuggingFaceTextGenInference
        HAS_NEW_HF = False
    from pydantic import BaseModel, Field
    HAS_LANGCHAIN = True
except ImportError:
    HAS_LANGCHAIN = False

# Try scikit-learn imports
try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import IsolationForest
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

logger = logging.getLogger("OpenWearables.HealthAnalyzer")

# Define Pydantic models for structured output
if HAS_LANGCHAIN:
    class CardiacAssessment(BaseModel):
        """Assessment of cardiac health."""
        heart_rate_status: str = Field(description="Status of heart rate (Normal, Bradycardia, Tachycardia)")
        hrv_status: str = Field(description="Status of heart rate variability (Normal, Low, High)")
        arrhythmia_detected: bool = Field(description="Whether an arrhythmia was detected")
        arrhythmia_type: Optional[str] = Field(None, description="Type of arrhythmia if detected")
        overall_assessment: str = Field(description="Overall assessment of cardiac health")
        confidence: float = Field(description="Confidence score for this assessment (0-1)")
    
    class RespiratoryAssessment(BaseModel):
        """Assessment of respiratory health."""
        spo2_status: str = Field(description="Status of blood oxygen saturation (Normal, Low, Critical)")
        respiratory_rate: Optional[float] = Field(None, description="Estimated respiratory rate (breaths per minute)")
        respiratory_status: str = Field(description="Overall respiratory status")
        confidence: float = Field(description="Confidence score for this assessment (0-1)")
    
    class HealthRecommendation(BaseModel):
        """Health recommendation."""
        category: str = Field(description="Category of recommendation (Exercise, Nutrition, Sleep, Medication, Medical)")
        recommendation: str = Field(description="Specific recommendation")
        urgency: str = Field(description="Urgency level (Routine, Important, Urgent)")
        explanation: str = Field(description="Explanation for the recommendation")
    
    class HealthAssessment(BaseModel):
        """Complete health assessment."""
        cardiac: CardiacAssessment = Field(description="Cardiac health assessment")
        respiratory: Optional[RespiratoryAssessment] = Field(None, description="Respiratory health assessment")
        activity_status: str = Field(description="Current activity status")
        stress_level: str = Field(description="Estimated stress level (Low, Moderate, High)")
        overall_status: str = Field(description="Overall health status summary")
        recommendations: List[HealthRecommendation] = Field(description="Health recommendations")

class HealthAnalyzer:
    """
    Analyzes health data and generates insights and recommendations.
    
    This class integrates traditional rule-based analysis with LLM-based
    analysis for comprehensive health assessment.
    """
    
    def __init__(self, config: Dict[str, Any], device: str = "cpu"):
        """
        Initialize the health analyzer.
        
        Args:
            config: Configuration dictionary
            device: Computing device for models
        """
        self.config = config
        self.device = device
        
        # Initialize LLM for health analysis if available
        self.health_llm = None
        self.llm_chain = None
        if HAS_LANGCHAIN:
            self._initialize_llm()
    
    def _initialize_llm(self) -> None:
        """Initialize language model for health analysis."""
        try:
            model_id = self.config.get("models", {}).get("health_assessment", "google/gemma-med-8b")
            logger.info(f"Initializing health LLM: {model_id}")
            
            # In a production environment, we'd use a local LLM server
            # For demo purposes, we'll simulate with HuggingFace endpoint
            if HAS_NEW_HF:
                self.health_llm = HuggingFaceEndpoint(
                    repo_id=model_id,
                    max_new_tokens=512,
                    top_k=10,
                    top_p=0.95,
                    temperature=0.1,
                    repetition_penalty=1.03
                )
            else:
                self.health_llm = HuggingFaceTextGenInference(
                    inference_server_url="http://localhost:8080/", # Placeholder URL
                    max_new_tokens=512,
                    top_k=10,
                    top_p=0.95,
                    temperature=0.1,
                    repetition_penalty=1.03
                )
            
            # Set up output parser for structured output
            output_parser = PydanticOutputParser(pydantic_object=HealthAssessment)
            
            # Create prompt template
            template = """
            You are a medical AI assistant analyzing wearable health sensor data.
            
            Analyze the following health data:
            
            Cardiac:
            - Heart Rate: {heart_rate} bpm
            - Heart Rate Variability: SDNN {hrv_sdnn} ms, RMSSD {hrv_rmssd} ms
            
            Respiratory:
            - Blood Oxygen (SpO2): {spo2}%
            
            Activity:
            - Current Activity: {activity}
            - Body Temperature: {temperature}°C
            
            User Information:
            - Age: {age} years
            - Gender: {gender}
            - Medical Conditions: {medical_conditions}
            - Medications: {medications}
            
            Provide a comprehensive health assessment and recommendations.
            
            {format_instructions}
            """
            
            prompt = PromptTemplate(
                template=template,
                input_variables=[
                    "heart_rate", "hrv_sdnn", "hrv_rmssd", "spo2", 
                    "activity", "temperature", "age", "gender", 
                    "medical_conditions", "medications"
                ],
                partial_variables={"format_instructions": output_parser.get_format_instructions()}
            )
            
            # Create chain using the new RunnableSequence syntax instead of deprecated LLMChain
            self.llm_chain = prompt | self.health_llm | output_parser
            logger.info("Health LLM initialized successfully")
        
        except Exception as e:
            logger.error(f"Error initializing health LLM: {str(e)}")
            self.health_llm = None
            self.llm_chain = None
    
    def analyze_health_data(self, processed_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze processed health data and generate insights.
        
        Args:
            processed_data: Dictionary of processed health metrics and features
            
        Returns:
            Dictionary of health analysis results
        """
        # Extract key health metrics
        heart_rate = processed_data.get("ecg", {}).get("heart_rate", 0)
        hrv_metrics = processed_data.get("ecg", {}).get("hrv", {})
        spo2 = processed_data.get("ppg", {}).get("spo2", 0)
        activity = processed_data.get("activity", "unknown")
        temperature = processed_data.get("temperature", 0)
        
        # Perform rule-based analysis first
        analysis_results = self._rule_based_analysis(
            heart_rate, hrv_metrics, spo2, activity, temperature
        )
        
        # Use LLM for enhanced analysis if available
        if self.health_llm and self.llm_chain:
            llm_results = self._llm_based_analysis(
                heart_rate, hrv_metrics, spo2, activity, temperature
            )
            
            # Combine rule-based and LLM-based analyses
            analysis_results.update(llm_results)
        
        return analysis_results
    
    def _rule_based_analysis(self, heart_rate: float, hrv_metrics: Dict[str, float], 
                             spo2: float, activity: str, temperature: float) -> Dict[str, Any]:
        """
        Perform rule-based analysis of health metrics.
        
        Args:
            heart_rate: Heart rate in bpm
            hrv_metrics: Dictionary of HRV metrics
            spo2: Blood oxygen saturation percentage
            activity: Current activity
            temperature: Body temperature in Celsius
            
        Returns:
            Dictionary of analysis results
        """
        results = {
            "cardiac_status": "normal",
            "respiratory_status": "normal",
            "stress_level": "low",
            "overall_status": "normal",
            "alerts": []
        }
        
        # Heart rate analysis
        if heart_rate < 50:
            results["cardiac_status"] = "bradycardia"
            results["alerts"].append({
                "type": "heart_rate",
                "severity": "moderate",
                "message": f"Low heart rate detected: {heart_rate} bpm"
            })
        elif heart_rate > 100:
            if activity == "resting":
                results["cardiac_status"] = "tachycardia"
                results["alerts"].append({
                    "type": "heart_rate",
                    "severity": "moderate",
                    "message": f"Elevated heart rate at rest: {heart_rate} bpm"
                })
            else:
                # Elevated heart rate during activity is normal
                results["cardiac_status"] = "normal"
        
        # HRV analysis
        sdnn = hrv_metrics.get("SDNN", 0)
        rmssd = hrv_metrics.get("RMSSD", 0)
        
        if sdnn < 20 or rmssd < 15:
            results["stress_level"] = "high"
            results["alerts"].append({
                "type": "stress",
                "severity": "moderate",
                "message": "Low heart rate variability indicates elevated stress"
            })
        
        # SpO2 analysis
        if spo2 < 90:
            results["respiratory_status"] = "hypoxic"
            results["alerts"].append({
                "type": "spo2",
                "severity": "high",
                "message": f"Low blood oxygen detected: {spo2}%"
            })
        elif spo2 < 95:
            results["respiratory_status"] = "borderline"
            results["alerts"].append({
                "type": "spo2",
                "severity": "moderate",
                "message": f"Blood oxygen slightly below normal: {spo2}%"
            })
        
        # Temperature analysis
        if temperature > 37.5:
            results["alerts"].append({
                "type": "temperature",
                "severity": "moderate",
                "message": f"Elevated body temperature: {temperature}°C"
            })
        elif temperature < 36.0:
            results["alerts"].append({
                "type": "temperature",
                "severity": "moderate",
                "message": f"Low body temperature: {temperature}°C"
            })
        
        # Overall status determination
        if any(alert["severity"] == "high" for alert in results["alerts"]):
            results["overall_status"] = "requires_attention"
        elif any(alert["severity"] == "moderate" for alert in results["alerts"]):
            results["overall_status"] = "caution"
        
        return results
    
    def _llm_based_analysis(self, heart_rate: float, hrv_metrics: Dict[str, float],
                            spo2: float, activity: str, temperature: float) -> Dict[str, Any]:
        """
        Perform LLM-based analysis of health metrics.
        
        Args:
            heart_rate: Heart rate in bpm
            hrv_metrics: Dictionary of HRV metrics
            spo2: Blood oxygen saturation percentage
            activity: Current activity
            temperature: Body temperature in Celsius
            
        Returns:
            Dictionary of LLM analysis results
        """
        try:
            # Prepare input data for LLM
            input_data = {
                "heart_rate": heart_rate,
                "hrv_sdnn": hrv_metrics.get("SDNN", 0),
                "hrv_rmssd": hrv_metrics.get("RMSSD", 0),
                "spo2": spo2,
                "activity": activity,
                "temperature": temperature,
                "age": 45,  # Default values for demo
                "gender": "unknown",
                "medical_conditions": "none",
                "medications": "none"
            }
            
            # Run LLM chain
            result = self.llm_chain.invoke(input_data)
            
            # For the demo, we'll simulate the LLM output
            if not isinstance(result, HealthAssessment):
                # Create a simulated response
                llm_results = {
                    "llm_analysis": {
                        "cardiac": {
                            "heart_rate_status": "Normal" if 60 <= heart_rate <= 100 else "Abnormal",
                            "hrv_status": "Normal" if hrv_metrics.get("SDNN", 0) >= 30 else "Low",
                            "arrhythmia_detected": False,
                            "overall_assessment": "Your cardiac parameters are within normal ranges."
                        },
                        "respiratory": {
                            "spo2_status": "Normal" if spo2 >= 95 else "Below normal",
                            "respiratory_status": "Your blood oxygen levels indicate normal respiratory function." if spo2 >= 95 else "Your blood oxygen levels are below normal range."
                        },
                        "recommendations": [
                            {
                                "category": "Activity",
                                "recommendation": "Maintain regular physical activity",
                                "urgency": "Routine",
                                "explanation": "Regular exercise helps maintain cardiovascular health."
                            }
                        ]
                    }
                }
            else:
                # Convert Pydantic model to dictionary
                llm_results = {"llm_analysis": result.dict()}
            
            return llm_results
        
        except Exception as e:
            logger.error(f"Error in LLM-based analysis: {str(e)}")
            return {"llm_analysis_error": str(e)}
    
    def generate_recommendations(self, user_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate personalized health recommendations.
        
        Args:
            user_data: Dictionary containing user health data and context
            
        Returns:
            Dictionary containing recommendations and explanations
        """
        if not self.health_llm:
            # Fallback to rule-based recommendations
            return self._generate_rule_based_recommendations(user_data)
        
        try:
            # Create a more detailed prompt for recommendations
            recommendation_template = """
            You are a medical AI assistant providing health recommendations.
            
            User Health Profile:
            - Age: {age} years
            - Gender: {gender}
            - Weight: {weight} kg
            - Height: {height} cm
            - Activity Level: {activity_level}
            - Medical Conditions: {medical_conditions}
            - Medications: {medications}
            
            Recent Health Data:
            - Average Heart Rate: {avg_heart_rate} bpm
            - Heart Rate Variability: {hrv}
            - Average SpO2: {avg_spo2}%
            - Sleep Duration: {sleep_duration} hours
            - Daily Steps: {steps}
            - Stress Level: {stress_level}
            
            Based on this health profile and data, provide 3-5 specific, actionable health recommendations.
            
            For each recommendation, include:
            1. A specific action the user should take
            2. Why this recommendation is important given their health data
            3. The expected benefit
            4. How to incorporate it into daily life
            
            Focus on evidence-based recommendations relevant to the user's specific health metrics.
            """
            
            # This would use a proper LLM chain in production
            # For the demo, we'll simulate a structured response
            
            recommendations = [
                {
                    "action": "Increase daily moderate exercise to 30 minutes",
                    "rationale": f"Your average heart rate of {user_data.get('avg_heart_rate', 75)} bpm and HRV metrics suggest you would benefit from more cardiovascular activity.",
                    "benefit": "Improved cardiovascular health, better stress management, and enhanced metabolic function.",
                    "implementation": "Add a brisk 30-minute walk during lunch break or after dinner."
                },
                {
                    "action": "Practice deep breathing exercises for 10 minutes daily",
                    "rationale": f"Your stress level is indicated as {user_data.get('stress_level', 'moderate')} based on HRV and other metrics.",
                    "benefit": "Reduced stress, improved heart rate variability, and better sleep quality.",
                    "implementation": "Use a guided breathing app in the morning and before bed."
                },
                {
                    "action": "Optimize hydration with 2-3 liters of water daily",
                    "rationale": "Proper hydration improves blood circulation and oxygen delivery.",
                    "benefit": "Enhanced cellular function, improved cognitive performance, and better temperature regulation.",
                    "implementation": "Keep a water bottle with you and set reminders to drink regularly throughout the day."
                }
            ]
            
            if user_data.get('avg_spo2', 98) < 96:
                recommendations.append({
                    "action": "Monitor oxygen levels more frequently and consult a healthcare provider",
                    "rationale": f"Your average SpO2 of {user_data.get('avg_spo2', 95)}% is slightly below the optimal range.",
                    "benefit": "Early detection of potential respiratory issues and appropriate medical intervention if needed.",
                    "implementation": "Use your wearable to check SpO2 3-4 times daily and record any symptoms."
                })
            
            if user_data.get('sleep_duration', 7) < 7:
                recommendations.append({
                    "action": "Increase sleep duration to 7-8 hours with consistent sleep/wake times",
                    "rationale": f"Your current sleep duration of {user_data.get('sleep_duration', 6)} hours is below recommendations for optimal health.",
                    "benefit": "Improved cognitive function, enhanced recovery, better immune function, and reduced stress.",
                    "implementation": "Set a consistent bedtime routine and limit screen time 1 hour before bed."
                })
            
            return {
                "personalized_recommendations": recommendations,
                "priority": "medium",
                "followup_timeframe": "2 weeks"
            }
        
        except Exception as e:
            logger.error(f"Error generating recommendations: {str(e)}")
            return self._generate_rule_based_recommendations(user_data)
    
    def _generate_rule_based_recommendations(self, user_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate rule-based health recommendations when LLM is unavailable.
        
        Args:
            user_data: Dictionary containing user health data
            
        Returns:
            Dictionary of basic recommendations
        """
        recommendations = []
        
        # Basic recommendations based on heart rate
        avg_heart_rate = user_data.get('avg_heart_rate', 75)
        if avg_heart_rate > 90:
            recommendations.append({
                "action": "Incorporate more relaxation activities in your daily routine",
                "rationale": "Your elevated average heart rate may indicate stress or insufficient recovery.",
                "benefit": "Reduced cardiovascular strain and improved overall well-being.",
                "implementation": "Try meditation, deep breathing, or gentle yoga for 15 minutes daily."
            })
        
        # Activity recommendations
        steps = user_data.get('steps', 5000)
        if steps < 7500:
            recommendations.append({
                "action": "Increase daily step count to at least 7,500",
                "rationale": "Regular movement improves cardiovascular health and metabolism.",
                "benefit": "Reduced risk of chronic disease and improved energy levels.",
                "implementation": "Take short walking breaks, use stairs instead of elevators, and park farther from entrances."
            })
        
        # Sleep recommendations
        sleep_duration = user_data.get('sleep_duration', 7)
        if sleep_duration < 7:
            recommendations.append({
                "action": "Aim for 7-8 hours of quality sleep each night",
                "rationale": "Sufficient sleep is essential for recovery and health.",
                "benefit": "Improved cognitive function, metabolism, and immune system.",
                "implementation": "Establish a consistent sleep schedule and create a relaxing bedtime routine."
            })
        
        # Hydration recommendation (universal)
        recommendations.append({
            "action": "Drink at least 2 liters of water daily",
            "rationale": "Proper hydration supports all bodily functions.",
            "benefit": "Improved energy, better skin health, and optimal organ function.",
            "implementation": "Keep a water bottle with you and drink consistently throughout the day."
        })
        
        return {
            "personalized_recommendations": recommendations,
            "priority": "medium",
            "followup_timeframe": "2 weeks"
        }
    
    def detect_anomalies(self, processed_data: Dict[str, Any], historical_data: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Detect anomalies in health data compared to historical baseline.
        
        Args:
            processed_data: Current processed health data
            historical_data: Historical health data for comparison
            
        Returns:
            List of detected anomalies with details
        """
        anomalies = []
        
        # If no historical data, can only do threshold-based detection
        if not historical_data:
            # Heart rate anomalies
            heart_rate = processed_data.get("ecg", {}).get("heart_rate", 0)
            if heart_rate > 0:
                if heart_rate < 40:
                    anomalies.append({
                        "type": "heart_rate",
                        "severity": "high",
                        "message": f"Severe bradycardia detected: {heart_rate} bpm",
                        "recommendation": "Seek immediate medical attention"
                    })
                elif heart_rate < 50:
                    anomalies.append({
                        "type": "heart_rate",
                        "severity": "medium",
                        "message": f"Bradycardia detected: {heart_rate} bpm",
                        "recommendation": "Monitor and consult healthcare provider if persistent"
                    })
                elif heart_rate > 150:
                    anomalies.append({
                        "type": "heart_rate",
                        "severity": "high",
                        "message": f"Severe tachycardia detected: {heart_rate} bpm",
                        "recommendation": "Seek immediate medical attention"
                    })
                elif heart_rate > 100 and processed_data.get("activity") == "resting":
                    anomalies.append({
                        "type": "heart_rate",
                        "severity": "medium",
                        "message": f"Tachycardia at rest detected: {heart_rate} bpm",
                        "recommendation": "Monitor and consult healthcare provider if persistent"
                    })
            
            # SpO2 anomalies
            spo2 = processed_data.get("ppg", {}).get("spo2", 0)
            if spo2 > 0:
                if spo2 < 90:
                    anomalies.append({
                        "type": "spo2",
                        "severity": "high",
                        "message": f"Severe hypoxemia detected: SpO2 {spo2}%",
                        "recommendation": "Seek immediate medical attention"
                    })
                elif spo2 < 95:
                    anomalies.append({
                        "type": "spo2",
                        "severity": "medium",
                        "message": f"Low blood oxygen detected: SpO2 {spo2}%",
                        "recommendation": "Monitor closely and consult healthcare provider if persistent"
                    })
            
            # Temperature anomalies
            temperature = processed_data.get("temperature", 0)
            if temperature > 0:
                if temperature > 38.0:
                    anomalies.append({
                        "type": "temperature",
                        "severity": "medium",
                        "message": f"Fever detected: {temperature}°C",
                        "recommendation": "Rest, hydrate, and monitor symptoms"
                    })
                elif temperature > 39.0:
                    anomalies.append({
                        "type": "temperature",
                        "severity": "high",
                        "message": f"High fever detected: {temperature}°C",
                        "recommendation": "Seek medical attention"
                    })
        
        else:
            # With historical data, we can detect deviations from personal baseline
            # Implementation would compare current values to historical averages
            # and detect significant deviations
            pass
        
        return anomalies


class HealthInsight:
    """
    Container class for health insights and recommendations.
    
    This class provides a structured way to store and present health
    analysis results, insights, and actionable recommendations.
    """
    
    def __init__(self, data: Dict[str, Any] = None):
        """
        Initialize HealthInsight container.
        
        Args:
            data: Dictionary containing analysis results and insights
        """
        self.data = data or {}
        self.timestamp = pd.Timestamp.now()
        self.version = "1.0"
    
    def get_overall_health_score(self) -> float:
        """
        Get overall health score from 0-100.
        
        Returns:
            Health score based on multiple factors
        """
        scores = []
        
        # Cardiac health score
        cardiac_analysis = self.data.get("cardiac_analysis", {})
        heart_rate_status = cardiac_analysis.get("heart_rate_status", "unknown")
        hrv_status = cardiac_analysis.get("hrv_status", "unknown")
        
        cardiac_score = 85.0  # Base score
        if heart_rate_status == "normal":
            cardiac_score += 10
        elif heart_rate_status in ["bradycardia", "tachycardia"]:
            cardiac_score -= 15
        
        if hrv_status == "high":
            cardiac_score += 5
        elif hrv_status == "low":
            cardiac_score -= 10
        
        scores.append(max(0, min(100, cardiac_score)))
        
        # Respiratory health score
        respiratory_analysis = self.data.get("respiratory_analysis", {})
        spo2_status = respiratory_analysis.get("spo2_status", "unknown")
        
        respiratory_score = 90.0  # Base score
        if spo2_status == "normal":
            respiratory_score += 10
        elif spo2_status == "low":
            respiratory_score -= 20
        elif spo2_status == "critical":
            respiratory_score -= 40
        
        scores.append(max(0, min(100, respiratory_score)))
        
        # Activity health score
        activity_status = self.data.get("activity_status", "unknown")
        activity_score = 80.0  # Base score
        
        if activity_status == "active":
            activity_score += 15
        elif activity_status == "sedentary":
            activity_score -= 10
        
        scores.append(max(0, min(100, activity_score)))
        
        # Stress level score
        stress_level = self.data.get("stress_level", "moderate")
        stress_score = 85.0  # Base score
        
        if stress_level == "low":
            stress_score += 10
        elif stress_level == "high":
            stress_score -= 20
        
        scores.append(max(0, min(100, stress_score)))
        
        # Calculate weighted average
        if scores:
            return round(sum(scores) / len(scores), 1)
        return 75.0  # Default score
    
    def get_risk_factors(self) -> List[Dict[str, Any]]:
        """
        Get identified health risk factors.
        
        Returns:
            List of risk factors with severity and recommendations
        """
        risk_factors = []
        
        # Check for anomalies
        anomalies = self.data.get("anomalies", [])
        for anomaly in anomalies:
            risk_factors.append({
                "factor": anomaly.get("type", "unknown"),
                "severity": anomaly.get("severity", "medium"),
                "description": anomaly.get("message", ""),
                "recommendation": anomaly.get("recommendation", "")
            })
        
        # Check cardiac risks
        cardiac_analysis = self.data.get("cardiac_analysis", {})
        if cardiac_analysis.get("heart_rate_status") in ["bradycardia", "tachycardia"]:
            risk_factors.append({
                "factor": "cardiac_arrhythmia",
                "severity": "medium",
                "description": f"Heart rate abnormality: {cardiac_analysis.get('heart_rate_status')}",
                "recommendation": "Monitor regularly and consult healthcare provider if persistent"
            })
        
        if cardiac_analysis.get("hrv_status") == "low":
            risk_factors.append({
                "factor": "poor_heart_rate_variability",
                "severity": "low",
                "description": "Low heart rate variability indicates reduced cardiovascular fitness",
                "recommendation": "Increase regular aerobic exercise and stress management practices"
            })
        
        # Check respiratory risks
        respiratory_analysis = self.data.get("respiratory_analysis", {})
        if respiratory_analysis.get("spo2_status") in ["low", "critical"]:
            risk_factors.append({
                "factor": "hypoxemia",
                "severity": "high" if respiratory_analysis.get("spo2_status") == "critical" else "medium",
                "description": f"Low blood oxygen saturation: {respiratory_analysis.get('spo2_status')}",
                "recommendation": "Seek medical evaluation for respiratory function"
            })
        
        # Check stress risks
        stress_level = self.data.get("stress_level", "moderate")
        if stress_level == "high":
            risk_factors.append({
                "factor": "chronic_stress",
                "severity": "medium",
                "description": "Elevated stress levels detected through physiological markers",
                "recommendation": "Implement stress reduction techniques and consider counseling"
            })
        
        return risk_factors
    
    def get_recommendations(self) -> List[Dict[str, Any]]:
        """
        Get health recommendations with priorities.
        
        Returns:
            List of recommendations sorted by priority
        """
        recommendations = self.data.get("personalized_recommendations", [])
        
        # Add priority levels if not present
        for rec in recommendations:
            if "priority" not in rec:
                # Determine priority based on health status
                if any(rf["severity"] == "high" for rf in self.get_risk_factors()):
                    rec["priority"] = "high"
                elif any(rf["severity"] == "medium" for rf in self.get_risk_factors()):
                    rec["priority"] = "medium"
                else:
                    rec["priority"] = "low"
        
        # Sort by priority
        priority_order = {"high": 3, "medium": 2, "low": 1}
        recommendations.sort(key=lambda x: priority_order.get(x.get("priority", "low"), 1), reverse=True)
        
        return recommendations
    
    def get_trends(self) -> Dict[str, Any]:
        """
        Get health trends and patterns.
        
        Returns:
            Dictionary of health trends
        """
        trends = self.data.get("trends", {})
        
        # Add basic trend analysis if not present
        if not trends:
            health_score = self.get_overall_health_score()
            
            trends = {
                "overall_health": {
                    "current_score": health_score,
                    "trend": "stable",  # Would be calculated from historical data
                    "change": 0.0
                },
                "cardiac_fitness": {
                    "trend": "improving" if health_score > 80 else "stable",
                    "change": 0.0
                },
                "stress_levels": {
                    "trend": "stable",
                    "change": 0.0
                },
                "activity_levels": {
                    "trend": "stable",
                    "change": 0.0
                }
            }
        
        return trends
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive health summary.
        
        Returns:
            Dictionary with complete health overview
        """
        return {
            "timestamp": self.timestamp.isoformat(),
            "health_score": self.get_overall_health_score(),
            "status": self.data.get("overall_status", "Normal"),
            "risk_factors_count": len(self.get_risk_factors()),
            "high_priority_recommendations": len([r for r in self.get_recommendations() if r.get("priority") == "high"]),
            "key_metrics": {
                "heart_rate": self.data.get("cardiac_analysis", {}).get("heart_rate", "N/A"),
                "spo2": self.data.get("respiratory_analysis", {}).get("spo2", "N/A"),
                "stress_level": self.data.get("stress_level", "N/A"),
                "activity_status": self.data.get("activity_status", "N/A")
            }
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert insight to dictionary format."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "version": self.version,
            "health_score": self.get_overall_health_score(),
            "risk_factors": self.get_risk_factors(),
            "recommendations": self.get_recommendations(),
            "trends": self.get_trends(),
            "summary": self.get_summary(),
            "raw_data": self.data
        }
    
    def export_report(self, filepath: str, format: str = "json") -> None:
        """
        Export health insight report to file.
        
        Args:
            filepath: Path to output file
            format: Export format ('json', 'csv', 'html')
        """
        if format.lower() == "json":
            import json
            with open(filepath, 'w') as f:
                json.dump(self.to_dict(), f, indent=2, default=str)
        
        elif format.lower() == "csv":
            df = pd.DataFrame([self.get_summary()])
            df.to_csv(filepath, index=False)
        
        elif format.lower() == "html":
            self._export_html_report(filepath)
        
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def _export_html_report(self, filepath: str) -> None:
        """Export HTML health report."""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Health Insight Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background-color: #f0f8ff; padding: 20px; border-radius: 10px; }}
                .score {{ font-size: 24px; font-weight: bold; color: #2e8b57; }}
                .section {{ margin: 20px 0; padding: 15px; border-left: 4px solid #4CAF50; }}
                .risk-high {{ color: #d32f2f; }}
                .risk-medium {{ color: #f57c00; }}
                .risk-low {{ color: #388e3c; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Health Insight Report</h1>
                <p>Generated: {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p class="score">Overall Health Score: {self.get_overall_health_score()}/100</p>
            </div>
            
            <div class="section">
                <h2>Risk Factors</h2>
                {"<p>No significant risk factors identified.</p>" if not self.get_risk_factors() else ""}
                {"".join([f'<p class="risk-{rf["severity"]}">{rf["description"]} - {rf["recommendation"]}</p>' for rf in self.get_risk_factors()])}
            </div>
            
            <div class="section">
                <h2>Recommendations</h2>
                {"".join([f'<p><strong>{rec["action"]}</strong><br>{rec["rationale"]}<br><em>Benefit: {rec["benefit"]}</em></p>' for rec in self.get_recommendations()])}
            </div>
        </body>
        </html>
        """
        
        with open(filepath, 'w') as f:
            f.write(html_content)
    
    def __repr__(self) -> str:
        """String representation of HealthInsight."""
        return f"HealthInsight(score={self.get_overall_health_score()}, risks={len(self.get_risk_factors())}, timestamp={self.timestamp})"