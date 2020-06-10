target_sources(mrchem-tests
  PRIVATE
  )

add_Catch_test(
  NAME cavity_function
  LABELS "cavity_function"
  )

add_Catch_test(
  NAME reaction_operator
  LABELS "reaction_operator"
  )
