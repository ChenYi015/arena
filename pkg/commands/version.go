// Copyright 2018 The Kubeflow Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//       http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package commands

import (
	"fmt"

	"github.com/kubeflow/arena"
	"github.com/spf13/cobra"
)

func NewVersionCmd(cliName string) *cobra.Command {
	var (
		short   bool
		verbose bool
	)
	versionCmd := cobra.Command{
		Use:   "version",
		Short: "Print version information",
		Run: func(cmd *cobra.Command, args []string) {
			version := arena.GetVersion()
			fmt.Printf("%s: %s\n", cliName, version)
			if short {
				return
			}
			fmt.Printf("  BuildDate: %s\n", version.BuildDate)
			fmt.Printf("  GitCommit: %s\n", version.GitCommit)
			fmt.Printf("  GitTreeState: %s\n", version.GitTreeState)
			if version.GitTag != "" {
				fmt.Printf("  GitTag: %s\n", version.GitTag)
			}
			fmt.Printf("  GoVersion: %s\n", version.GoVersion)
			fmt.Printf("  Compiler: %s\n", version.Compiler)
			fmt.Printf("  Platform: %s\n", version.Platform)
			if verbose {
				fmt.Printf("  ChartsHome: %s\n", version.ChartsInfo.ChartsHome)
				fmt.Printf("  ChartsVersion:\n")
				for chartName, chartVersion := range version.ChartsInfo.ChartsVersion {
					fmt.Printf("    %s: %s\n", chartName, chartVersion)
				}
				// Display CRD versions
				fmt.Printf("  CRDVersions:\n")
				if len(version.CRDVersions) == 0 {
					fmt.Printf("    (Run 'arena version -v' with cluster access to see CRD versions)\n")
				} else {
					for crdName, crdVersion := range version.CRDVersions {
						fmt.Printf("    %s: %s\n", crdName, crdVersion)
					}
				}
				// Display operator versions
				fmt.Printf("  OperatorVersions:\n")
				if len(version.OperatorVersion) == 0 {
					fmt.Printf("    (Run 'arena version -v' with cluster access to see operator versions)\n")
				} else {
					for operatorType, operatorVersion := range version.OperatorVersion {
						fmt.Printf("    %s: %s\n", operatorType, operatorVersion)
					}
				}
			}
		},
	}
	versionCmd.Flags().BoolVar(&short, "short", false, "print just the version number")
	versionCmd.Flags().BoolVarP(&verbose, "verbose", "v", false, "print the supported charts version number, CRD versions, and operator versions deployed in cluster")
	return &versionCmd
}
