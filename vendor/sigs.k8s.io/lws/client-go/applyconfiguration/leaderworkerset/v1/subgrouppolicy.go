/*
Copyright 2023.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
// Code generated by applyconfiguration-gen. DO NOT EDIT.

package v1

// SubGroupPolicyApplyConfiguration represents an declarative configuration of the SubGroupPolicy type for use
// with apply.
type SubGroupPolicyApplyConfiguration struct {
	SubGroupSize *int32 `json:"subGroupSize,omitempty"`
}

// SubGroupPolicyApplyConfiguration constructs an declarative configuration of the SubGroupPolicy type for use with
// apply.
func SubGroupPolicy() *SubGroupPolicyApplyConfiguration {
	return &SubGroupPolicyApplyConfiguration{}
}

// WithSubGroupSize sets the SubGroupSize field in the declarative configuration to the given value
// and returns the receiver, so that objects can be built by chaining "With" function invocations.
// If called multiple times, the SubGroupSize field is set to the value of the last call.
func (b *SubGroupPolicyApplyConfiguration) WithSubGroupSize(value int32) *SubGroupPolicyApplyConfiguration {
	b.SubGroupSize = &value
	return b
}
