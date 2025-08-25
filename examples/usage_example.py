"""
DS Assistant Agent Framework - Usage Example

This example demonstrates how to use the flexible DS Assistant Agent Framework
with custom catalog.schema configuration.
"""

from ds_assistant.agents.ds_agent_framework import create_ds_assistant_agent, deploy_ds_assistant_agent

def main():
    # Example 1: Default configuration
    print("=== Example 1: Default Configuration ===")
    agent_default = create_ds_assistant_agent()
    print(f"Created agent with default config: main.analytics")
    
    # Example 2: Custom catalog.schema
    print("\n=== Example 2: Custom Configuration ===")
    agent_custom = create_ds_assistant_agent(
        llm_endpoint_name="databricks-claude-3-5-sonnet",
        catalog="my_catalog",
        schema="ml_analytics",
        auto_create_functions=True
    )
    print(f"Created agent with custom config: my_catalog.ml_analytics")
    
    # Example 3: Without auto-creating functions (manual setup)
    print("\n=== Example 3: Manual Function Setup ===")
    agent_manual = create_ds_assistant_agent(
        catalog="existing_catalog",
        schema="existing_schema",
        auto_create_functions=False  # Functions already exist
    )
    print(f"Created agent with existing functions: existing_catalog.existing_schema")
    
    # Example 4: Test the agent
    print("\n=== Example 4: Testing the Agent ===")
    test_query = {
        "messages": [
            {
                "role": "user",
                "content": "Analyze the table 'my_catalog.ml_analytics.sample_data' and recommend the best modeling strategy for predicting 'target_column'"
            }
        ]
    }
    
    # This would be the actual test (commented out for demo)
    # response = agent_custom.predict(test_query)
    # print("Agent response:", response.messages[-1].content[0].text)
    print("Agent ready for testing!")
    
    # Example 5: Deployment (commented out for demo)
    print("\n=== Example 5: Deployment ===")
    print("To deploy the agent:")
    print("""
    deployment_info = deploy_ds_assistant_agent(
        agent=agent_custom,
        model_name="my_ds_agent",
        catalog="my_catalog",
        schema="ml_analytics",
        llm_endpoint_name="databricks-claude-3-5-sonnet"
    )
    """)

if __name__ == "__main__":
    main()
