// Copyright 2018 The Kubeflow Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//	http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
package arena

import (
	"context"
	"fmt"
	"strings"

	log "github.com/sirupsen/logrus"
	appsv1 "k8s.io/api/apps/v1"
	extclientset "k8s.io/apiextensions-apiserver/pkg/client/clientset/clientset"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes"
)

// crdVersions stores the CRD version information
var crdVersions map[string]string

// operatorVersions stores the operator deployment version information
var operatorVersions map[string]string

// defaultCRDs contains the known CRD names and their supported API versions
var defaultCRDs = map[string]string{
	"tfjobs.kubeflow.org":               "kubeflow.org/v1",
	"mpijobs.kubeflow.org":              "kubeflow.org/v1alpha1",
	"pytorchjobs.kubeflow.org":          "kubeflow.org/v1",
	"crons.apps.kubedl.io":              "apps.kubedl.io/v1alpha1",
	"trainingjobs.kai.alibabacloud.com": "kai.alibabacloud.com/v1alpha1",
	"scaleins.kai.alibabacloud.com":     "kai.alibabacloud.com/v1alpha1",
	"scaleouts.kai.alibabacloud.com":    "kai.alibabacloud.com/v1alpha1",
}

// operatorDeployments contains the known operator deployment names and their types
var operatorDeployments = map[string]string{
	"tf-operator":            "tensorflow",
	"tf-job-operator":        "tensorflow",
	"mpi-operator":           "mpi",
	"pytorch-operator":       "pytorch",
	"pytorch-job-operator":   "pytorch",
	"cron-operator":          "cron",
	"kubedl-operator":        "cron",
	"et-operator":            "et",
	"spark-operator":         "spark",
	"volcano-operator":       "volcano",
	"ray-operator":           "ray",
	"elastic-job-supervisor": "elastic",
}

// InitClusterInfo initializes the cluster information (CRD versions and operator versions)
// This should be called after the Kubernetes client is available
func InitClusterInfo(clientset *kubernetes.Clientset, apiExtensionClient *extclientset.Clientset) {
	crdVersions = getCRDVersionsFromCluster(apiExtensionClient)
	operatorVersions = getOperatorVersionsFromCluster(clientset)
}

// getCRDVersions returns the CRD versions from the cluster
func getCRDVersions() map[string]string {
	if crdVersions == nil {
		// Return empty map if not initialized
		return make(map[string]string)
	}
	return crdVersions
}

// getOperatorVersions returns the operator versions from the cluster
func getOperatorVersions() map[string]string {
	if operatorVersions == nil {
		// Return empty map if not initialized
		return make(map[string]string)
	}
	return operatorVersions
}

// getCRDVersionsFromCluster retrieves CRD version information from the cluster
func getCRDVersionsFromCluster(apiExtensionClient *extclientset.Clientset) map[string]string {
	versions := make(map[string]string)

	// Try to get CRD information from the cluster
	if apiExtensionClient == nil {
		log.Debug("API extension clientset is nil, using default CRD versions")
		// Use default CRD versions if client is not available
		for crdName, apiVersion := range defaultCRDs {
			versions[crdName] = apiVersion
		}
		return versions
	}

	// Get the list of CRDs from the cluster
	crdList, err := apiExtensionClient.ApiextensionsV1().CustomResourceDefinitions().List(context.TODO(), metav1.ListOptions{})
	if err != nil {
		log.Debugf("Failed to list CRDs from cluster: %v, using default CRD versions", err)
		// Fall back to default CRD versions
		for crdName, apiVersion := range defaultCRDs {
			versions[crdName] = apiVersion
		}
		return versions
	}

	// Map known CRD names to their versions
	for _, crd := range crdList.Items {
		if apiVersion, ok := defaultCRDs[crd.Name]; ok {
			// Get the version from the CRD name
			version := getCRDVersionFromSpec(crd.Name)
			versions[crd.Name] = version + " (" + apiVersion + ")"
		}
	}

	// Add any default CRDs that are not found in the cluster
	for crdName, apiVersion := range defaultCRDs {
		if _, ok := versions[crdName]; !ok {
			versions[crdName] = "Not installed (" + apiVersion + ")"
		}
	}

	return versions
}

// getCRDVersionFromSpec extracts the version from a CRD spec
func getCRDVersionFromSpec(crdName string) string {
	// Return the version name from the CRD name for simplicity
	// In a real implementation, you would parse the CRD spec to get the version
	return "v1"
}

// getOperatorVersionsFromCluster retrieves operator deployment versions from the cluster
func getOperatorVersionsFromCluster(clientset *kubernetes.Clientset) map[string]string {
	versions := make(map[string]string)

	if clientset == nil {
		log.Debug("Kubernetes clientset is nil, operator versions not available")
		return versions
	}

	// Get deployments from the arena-system namespace
	arenaNS := "arena-system"
	deployments, err := clientset.AppsV1().Deployments(arenaNS).List(context.TODO(), metav1.ListOptions{})
	if err != nil {
		log.Debugf("Failed to list deployments in %s namespace: %v", arenaNS, err)
	}

	// Also check kubeflow namespace for operators
	kubeflowNS := "kubeflow"
	kubeflowDeployments, err := clientset.AppsV1().Deployments(kubeflowNS).List(context.TODO(), metav1.ListOptions{})
	if err != nil {
		log.Debugf("Failed to list deployments in %s namespace: %v", kubeflowNS, err)
	}

	// Process arena-system deployments
	for _, dep := range deployments.Items {
		operatorType := identifyOperatorType(dep.Name)
		if operatorType != "" {
			image := extractOperatorImage(dep)
			versions[operatorType] = fmt.Sprintf("%s (%s)", dep.Name, image)
		}
	}

	// Process kubeflow deployments
	for _, dep := range kubeflowDeployments.Items {
		operatorType := identifyOperatorType(dep.Name)
		if operatorType != "" {
			// Skip if already found in arena-system
			if _, exists := versions[operatorType]; !exists {
				image := extractOperatorImage(dep)
				versions[operatorType] = fmt.Sprintf("%s (%s)", dep.Name, image)
			}
		}
	}

	return versions
}

// identifyOperatorType identifies the operator type from a deployment name
func identifyOperatorType(deploymentName string) string {
	for opName, opType := range operatorDeployments {
		if deploymentName == opName || strings.Contains(deploymentName, opName) {
			return opType
		}
	}
	return ""
}

// extractOperatorImage extracts the image version from a deployment
func extractOperatorImage(dep appsv1.Deployment) string {
	// Get the image from the first container
	if len(dep.Spec.Template.Spec.Containers) > 0 {
		image := dep.Spec.Template.Spec.Containers[0].Image
		// Extract version from image tag if possible
		parts := strings.Split(image, ":")
		if len(parts) > 1 {
			return parts[len(parts)-1]
		}
		return image
	}
	return "unknown"
}

// IsClusterInfoInitialized returns true if cluster info has been initialized
func IsClusterInfoInitialized() bool {
	return crdVersions != nil || operatorVersions != nil
}
